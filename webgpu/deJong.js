async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice({
    requiredFeatures: [["timestamp-query"]],
  });
  if (!device) {
    alert("need a browser that supports WebGPU");
    return;
  }

  const canvas = document.querySelector("canvas");
  const renderContext = canvas.getContext("webgpu");
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  renderContext.configure({
    device,
    format: presentationFormat,
  });

  const shadertext = await fetch("shader.wgsl").then((r) => r.text());
  const shader = device.createShaderModule({ code: shadertext });

  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shader,
    },
  });

  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    primitive: {
      topology: "triangle-strip",
    },
    vertex: {
      module: shader,
    },
    fragment: {
      module: shader,
      targets: [{ format: presentationFormat }],
    },
  });

  const renderPassDescriptor = {
    colorAttachments: [
      {
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  };

  const numUniforms = 7;
  const uniformBuffer = device.createBuffer({
    size: numUniforms * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformValues = new Float32Array(numUniforms);
  
  let redBuffer, greenBuffer, blueBuffer, computeBindGroup, renderBindGroup;
  function createBuffers(bufsize)
  {
    if(redBuffer) redBuffer.destroy();
    redBuffer = device.createBuffer({
      size: bufsize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if(greenBuffer) greenBuffer.destroy();
    greenBuffer = device.createBuffer({
      size: bufsize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if(blueBuffer) blueBuffer.destroy();
    blueBuffer = device.createBuffer({
      size: bufsize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    computeBindGroup = device.createBindGroup({
      label: "computeBindGroup",
      layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: redBuffer } },
          { binding: 2, resource: { buffer: greenBuffer } },
          { binding: 3, resource: { buffer: blueBuffer } },
        ],
    });
    renderBindGroup = device.createBindGroup({
      label: "renderBindGroup",
      layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 4, resource: { buffer: redBuffer } },
          { binding: 5, resource: { buffer: greenBuffer } },
          { binding: 6, resource: { buffer: blueBuffer } },
        ],
    });
  }
  
  createBuffers(4);

  const observer = new ResizeObserver((entries) => {
    assert(entries.length == 1);
    let entry = entries[0];
    const canvas = entry.target;
    const width = Math.floor(entry.contentBoxSize[0].inlineSize);
    const height = Math.floor(entry.contentBoxSize[0].blockSize);
    canvas.width = Math.max(
      1,
      Math.min(width, device.limits.maxTextureDimension2D)
    );
    canvas.height = Math.max(
      1,
      Math.min(height, device.limits.maxTextureDimension2D)
    );
    
    createBuffers(width * height * 4);
  });
  observer.observe(canvas);

  let frame = 0;
  let t = Math.random() * 1e4;
  let then = 0;
  let numPoints = 1e6;
  let ctime = 16;

  const computeTiming = new TimingHelper(device);
  const renderTiming = new TimingHelper(device);
  const computeAverage = new RollingAverage(120);
  const renderAverage = new RollingAverage(120);
  const frameAverage = new RollingAverage(120);
  const jsAverage = new RollingAverage();
  const infoElem = document.querySelector("#info");

  function render(now) {
    const deltaTime = now - then;
    then = now;

    const startTime = performance.now();
    
    let ratio = ctime / deltaTime;
    if (ratio > 0.7) numPoints /= 1.01;
    else if (ratio < 0.5) numPoints *= 1.01;

    uniformValues[0] = 4 * Math.sin(t * 1.03);
    uniformValues[1] = 4 * Math.sin(t * 1.07);
    uniformValues[2] = 4 * Math.sin(t * 1.09);
    uniformValues[3] = 4 * Math.sin(t * 1.13);
    uniformValues[4] = frame;
    uniformValues[5] = canvas.clientWidth;
    uniformValues[6] = numPoints;
    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    frame++;
    t += 1e-4;

    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(redBuffer);
    encoder.clearBuffer(greenBuffer);
    encoder.clearBuffer(blueBuffer);

    const computePass = computeTiming.beginComputePass(encoder);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(numPoints / 64 / 1024, 1);
    computePass.end();

    const view = renderContext.getCurrentTexture().createView();
    renderPassDescriptor.colorAttachments[0].view = view;
    const renderPass = renderTiming.beginRenderPass(
      encoder,
      renderPassDescriptor
    );
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(4);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    renderTiming.getResult().then((renderTime) => {
      renderAverage.addSample(renderTime / 1000000);
    });
    computeTiming.getResult().then((computeTime) => {
      ctime = computeTime / 1000000;
      computeAverage.addSample(computeTime / 1000000);
    });

    frameAverage.addSample(deltaTime);
    
    infoElem.textContent = `\
frame:   ${frameAverage.get().toFixed(2)}ms\
 (${(1000 / frameAverage.get()).toFixed(1)}fps)
js:      ${jsAverage.get().toFixed(2)}ms
compute: ${computeAverage.get().toFixed(2)}ms
render:  ${renderAverage.get().toFixed(2)}ms
points:  ${(numPoints/1e6).toFixed(1)}M`;

    const jsTime = performance.now() - startTime;
    jsAverage.addSample(jsTime);


    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

main();