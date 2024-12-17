export type DownloadInfo = {
  flight: number;
  staging: number;
};

type MeasureDefinition = {
  begin?: number;
  end?: number;
};

type RenderPassDefinition = {
  pipeline: GPURenderPipeline;
  bindings: Map<string, number>;
  measure: MeasureDefinition;
  descriptor: GPURenderPassDescriptor;
  vertices: number;
  bindGroup: GPUBindGroup | null;
};

type ComputePassDefinition = {
  pipeline: GPUComputePipeline;
  bindings: Map<string, number>;
  measure: MeasureDefinition;
  descriptor: GPUComputePassDescriptor;
  invocations: number | string;
  bindGroup: GPUBindGroup | null;
};

type DownloadPassDefinition = {
  buffer: string;
  maxFlight: number;
  inFlight: number;
  callback: (data: ArrayBuffer) => void;
  stagingQueue: Array<GPUBuffer>;
  stagingBuffer: GPUBuffer | null;
};

type PassDefinition = {
  compute?: ComputePassDefinition;
  render?: RenderPassDefinition;
  download?: DownloadPassDefinition;
};

export class WebGPU {
  private device: GPUDevice | null = null;
  private renderContext: GPUCanvasContext | null = null;
  private presentationFormat: GPUTextureFormat | null = null;
  private buffers = new Map<string, GPUBuffer>();
  private passes = new Array<PassDefinition>();
  private tsQuerySet: GPUQuerySet | null = null;

  private constructor() {}

  public static async create(canvas: HTMLCanvasElement, numts: number) {
    const webgpu = new WebGPU();
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (adapter === null) {
      throw `unable to aquire webgpu adapter`;
    }
    webgpu.device = await adapter.requestDevice({
      requiredFeatures: ['timestamp-query'],
    });
    if (webgpu.device === null) {
      throw `unable to aquire webgpu device`;
    }
    webgpu.renderContext = canvas.getContext('webgpu');
    if (webgpu.renderContext === null) {
      throw `unable to aquire webgpu context`;
    }
    webgpu.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    const device = webgpu.device;
    webgpu.renderContext.configure({
      device,
      format: webgpu.presentationFormat,
    });
    if (numts > 0) {
      webgpu.tsQuerySet = device.createQuerySet({
        type: 'timestamp',
        count: numts,
      });
    }
    return webgpu;
  }

  async shader(shaderfile: string) {
    if (this.device === null) throw 'no device';
    const shadertext = await fetch(shaderfile).then((r) => r.text());
    return this.device.createShaderModule({ code: shadertext });
  }

  public createBuffer(name: string, size: number, usage: GPUBufferUsageFlags) {
    if (this.device === null) throw 'no device';
    const old = this.buffers.get(name);
    if (old) old.destroy();
    this.buffers.set(
      name,
      this.device.createBuffer({
        size: size * 4,
        usage: usage,
      })
    );
    for (const pass of this.passes) {
      if ((pass.compute && pass.compute.bindings.has(name)) || (pass.render && pass.render.bindings.has(name))) {
        this.updateBindGroup(pass);
      }
    }
  }

  private updateBindGroup(pass: PassDefinition) {
    if (this.device === null) throw 'no device';
    const bindPass = pass.compute ? pass.compute : pass.render ? pass.render : null;
    if (bindPass) {
      const entries = new Array<GPUBindGroupEntry>();
      let missing = false;
      for (const [name, location] of bindPass.bindings) {
        const buffer = this.buffers.get(name);
        if (buffer) {
          entries.push({ binding: location, resource: { buffer: buffer } });
        } else {
          missing = true;
        }
      }
      if (!missing) {
        bindPass.bindGroup = this.device.createBindGroup({
          layout: bindPass.pipeline.getBindGroupLayout(0),
          entries,
        });
      }
    }
  }

  public addComputePass(
    shader: GPUShaderModule,
    entry: string,
    invocations: string | number,
    bindings: Map<string, number>,
    measure: MeasureDefinition = {}
  ) {
    if (this.device === null) throw 'no device';
    if (this.tsQuerySet === null) throw 'no ts query set';
    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shader,
        entryPoint: entry,
      },
    });
    const passDescriptor: GPUComputePassDescriptor = {};
    if (Object.keys(measure).length > 0) {
      passDescriptor.timestampWrites = { querySet: this.tsQuerySet };
      if (measure.begin !== undefined) {
        passDescriptor.timestampWrites.beginningOfPassWriteIndex = measure.begin;
      }
      if (measure.end !== undefined) {
        passDescriptor.timestampWrites.endOfPassWriteIndex = measure.end;
      }
    }
    const pass = {
      compute: {
        pipeline: pipeline,
        bindings: bindings,
        measure: measure,
        descriptor: passDescriptor,
        invocations: invocations,
        bindGroup: null,
      },
    };
    this.updateBindGroup(pass);
    this.passes.push(pass);
  }

  public addRenderPass(
    shader: GPUShaderModule,
    vsentry: string,
    fsentry: string,
    vertices: number,
    bindings: Map<string, number>,
    measure: MeasureDefinition = {}
  ) {
    if (this.device === null) throw 'no device';
    if (this.tsQuerySet === null) throw 'no ts query set';
    if (this.presentationFormat === null) throw 'no presentation format';
    if (this.renderContext === null) throw 'no render context';
    const pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shader,
        entryPoint: vsentry,
      },
      fragment: {
        module: shader,
        entryPoint: fsentry,
        targets: [{ format: this.presentationFormat }],
      },
    });
    const passDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          loadOp: 'clear',
          storeOp: 'store',
          view: this.renderContext.getCurrentTexture().createView(),
        },
      ],
    };
    if (Object.keys(measure).length > 0) {
      passDescriptor.timestampWrites = { querySet: this.tsQuerySet };
      if (measure.begin !== undefined) {
        passDescriptor.timestampWrites.beginningOfPassWriteIndex = measure.begin;
      }
      if (measure.end !== undefined) {
        passDescriptor.timestampWrites.endOfPassWriteIndex = measure.end;
      }
    }
    const pass: PassDefinition = {
      render: {
        pipeline: pipeline,
        bindings: bindings,
        measure: measure,
        descriptor: passDescriptor,
        vertices: vertices,
        bindGroup: null,
      },
    };
    this.updateBindGroup(pass);
    this.passes.push(pass);
  }

  public addBufferDownload(name: string, maxflight: number, callback: (arrayBuffer: ArrayBuffer) => void) {
    const pass: PassDefinition = {
      download: {
        buffer: name,
        maxFlight: maxflight,
        inFlight: 0,
        callback: callback,
        stagingQueue: [],
        stagingBuffer: null,
      },
    };
    this.passes.push(pass);
  }

  public updateBuffer(name: string, values: BufferSource) {
    if (this.device === null) throw 'no device';
    const buffer = this.buffers.get(name);
    if (buffer) this.device.queue.writeBuffer(buffer, 0, values);
  }

  public execute() {
    if (this.device === null) throw 'no device';
    if (this.tsQuerySet === null) throw 'no ts query set';
    if (this.renderContext === null) throw 'no render context';
    const encoder = this.device.createCommandEncoder();
    const buffer = this.buffers.get('data'); // TODO: remove hardcoded clear of data buffer. API?
    if (buffer) encoder.clearBuffer(buffer);
    for (const passDef of this.passes) {
      if (passDef.compute) {
        const pass = encoder.beginComputePass(passDef.compute.descriptor);
        pass.setPipeline(passDef.compute.pipeline);
        pass.setBindGroup(0, passDef.compute.bindGroup);
        if (typeof passDef.compute.invocations === 'number' && Number.isInteger(passDef.compute.invocations)) {
          pass.dispatchWorkgroups(passDef.compute.invocations);
        } else if (typeof passDef.compute.invocations === 'string') {
          const buffer = this.buffers.get(passDef.compute.invocations);
          if (buffer) pass.dispatchWorkgroupsIndirect(buffer, 0);
        }
        pass.end();
        if (Object.keys(passDef.compute.measure).length > 0) {
          const buffer = this.buffers.get('timestamp'); // TODO: remove hardcoded target buffer. API?
          if (buffer) encoder.resolveQuerySet(this.tsQuerySet, 0, this.tsQuerySet.count, buffer, 0);
        }
      } else if (passDef.render) {
        const view = this.renderContext.getCurrentTexture().createView();
        const colorAtt0 = [...passDef.render.descriptor.colorAttachments][0];
        if (colorAtt0) {
          colorAtt0.view = view;
        }
        const pass = encoder.beginRenderPass(passDef.render.descriptor);
        pass.setPipeline(passDef.render.pipeline);
        pass.setBindGroup(0, passDef.render.bindGroup);
        pass.draw(passDef.render.vertices);
        pass.end();
        if (Object.keys(passDef.render.measure).length > 0) {
          const buffer = this.buffers.get('timestamp'); // TODO: remove hardcoded target buffer. API?
          if (buffer) encoder.resolveQuerySet(this.tsQuerySet, 0, this.tsQuerySet.count, buffer, 0);
        }
      } else if (passDef.download) {
        passDef.download.stagingBuffer = null;
        const src = this.buffers.get(passDef.download.buffer);
        if (src) {
          if (passDef.download.inFlight < passDef.download.maxFlight) {
            const stagingBuffer =
              passDef.download.stagingQueue.pop() ||
              this.device.createBuffer({
                size: src.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
              });
            passDef.download.inFlight++;
            encoder.copyBufferToBuffer(src, 0, stagingBuffer, 0, stagingBuffer.size);
            passDef.download.stagingBuffer = stagingBuffer;
          }
        }
      }
    }
    const commandBuffer = encoder.finish();
    this.device.queue.submit([commandBuffer]);
    for (const passDef of this.passes) {
      if (passDef.download && passDef.download.stagingBuffer) {
        const download = passDef.download;
        const buffer = passDef.download.stagingBuffer;
        buffer.mapAsync(GPUMapMode.READ).then(() => {
          download.callback(buffer.getMappedRange());
          buffer.unmap();
          download.stagingQueue.push(buffer);
          download.inFlight--;
        });
      }
    }
  }

  public downloadInfo() {
    let staging = 0;
    let flight = 0;
    for (const passDef of this.passes) {
      if (passDef.download) {
        staging += passDef.download.stagingQueue.length;
        flight += passDef.download.inFlight;
      }
    }
    return { flight: flight, staging: staging };
  }
}
