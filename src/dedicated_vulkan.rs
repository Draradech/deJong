//! Import a buffer with dedicated device-local Vulkan memory into wgpu.

use ash::vk;
use std::error::Error;

pub fn can_import(desc: &wgpu::BufferDescriptor<'_>) -> bool {
    // Raw managed buffers cannot be mapped. vkCmdFillBuffer initializes complete
    // 4-byte words, so leave buffers with a partial final word to wgpu as well.
    !desc.mapped_at_creation
        && desc.size > 0
        && desc.size.is_multiple_of(4)
        && !desc.usage.intersects(
            wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::MAP_WRITE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::TLAS_INPUT,
        )
}

fn vk_usage(usage: wgpu::BufferUsages) -> vk::BufferUsageFlags {
    // TRANSFER_DST is also used for the required initialization before import.
    let mut flags = vk::BufferUsageFlags::TRANSFER_DST;
    if usage.contains(wgpu::BufferUsages::COPY_SRC) {
        flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(wgpu::BufferUsages::INDEX) {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(wgpu::BufferUsages::VERTEX) {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(wgpu::BufferUsages::UNIFORM) {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(wgpu::BufferUsages::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(wgpu::BufferUsages::INDIRECT) {
        flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    flags
}

pub fn create_buffer(device: &wgpu::Device, desc: &wgpu::BufferDescriptor<'_>) -> Result<wgpu::Buffer, Box<dyn Error>> {
    // The caller selects this only for a native Vulkan device. The HAL guard keeps
    // the underlying device alive while the raw buffer is created and imported.
    let hal =
        unsafe { device.as_hal::<wgpu_hal::api::Vulkan>() }.ok_or("dedicated buffer requires a Vulkan wgpu device")?;
    let shared = hal.shared_instance();
    let physical_api_version =
        unsafe { shared.raw_instance().get_physical_device_properties(hal.raw_physical_device()).api_version };
    if shared.instance_api_version() < vk::API_VERSION_1_1 || physical_api_version < vk::API_VERSION_1_1 {
        return Err("dedicated buffers require Vulkan 1.1".into());
    }
    let raw_device = hal.raw_device();
    assert!(can_import(desc));
    let usage = vk_usage(desc.usage);
    let create_info =
        vk::BufferCreateInfo::default().size(desc.size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { raw_device.create_buffer(&create_info, None)? };
    let mut memory = vk::DeviceMemory::null();
    let mut pool = vk::CommandPool::null();

    let result = (|| -> Result<u64, Box<dyn Error>> {
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(buffer) };
        let properties =
            unsafe { shared.raw_instance().get_physical_device_memory_properties(hal.raw_physical_device()) };
        let memory_type = (0..properties.memory_type_count)
            .find(|&index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && properties.memory_types[index as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or("no compatible device-local Vulkan memory type")?;

        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut dedicated);
        memory = unsafe { raw_device.allocate_memory(&allocate_info, None)? };
        unsafe { raw_device.bind_buffer_memory(buffer, memory, 0)? };

        // create_buffer_from_hal requires initialized memory. Fill before wgpu owns it.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(hal.queue_family_index())
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        pool = unsafe { raw_device.create_command_pool(&pool_info, None)? };
        let command_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let commands = unsafe { raw_device.allocate_command_buffers(&command_info)? };
        let command = commands[0];
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            raw_device.begin_command_buffer(command, &begin_info)?;
            raw_device.cmd_fill_buffer(command, buffer, 0, desc.size, 0);
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .buffer(buffer)
                .offset(0)
                .size(desc.size);
            raw_device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
            raw_device.end_command_buffer(command)?;
            let submission = vk::SubmitInfo::default().command_buffers(&commands);
            raw_device.queue_submit(hal.raw_queue(), &[submission], vk::Fence::null())?;
            raw_device.queue_wait_idle(hal.raw_queue())?;
        }
        Ok(requirements.size)
    })();

    if pool != vk::CommandPool::null() {
        unsafe { raw_device.destroy_command_pool(pool, None) };
    }
    let allocation_size = match result {
        Ok(size) => size,
        Err(error) => {
            unsafe {
                if memory != vk::DeviceMemory::null() {
                    raw_device.free_memory(memory, None);
                }
                raw_device.destroy_buffer(buffer, None);
            }
            return Err(error);
        }
    };

    // wgpu-hal owns and destroys both handles after the wgpu buffer is dropped.
    let hal_buffer = unsafe { wgpu_hal::vulkan::Buffer::from_raw_managed(buffer, memory, 0, allocation_size) };
    Ok(unsafe { device.create_buffer_from_hal::<wgpu_hal::api::Vulkan>(hal_buffer, desc) })
}
