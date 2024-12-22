export class WebGPU {
    constructor() {
        this.device = null;
        this.renderContext = null;
        this.presentationFormat = null;
        this.buffers = new Map();
        this.queries = new Map();
        this.passes = new Array();
    }
    static async create(canvas) {
        const webgpu = new WebGPU();
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
        });
        if (adapter === null)
            throw `unable to aquire webgpu adapter`;
        webgpu.device = await adapter.requestDevice({
            requiredFeatures: ['timestamp-query'],
        });
        if (webgpu.device === null)
            throw `unable to aquire webgpu device`;
        webgpu.renderContext = canvas.getContext('webgpu');
        if (webgpu.renderContext === null)
            throw `unable to aquire webgpu context`;
        webgpu.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        const device = webgpu.device;
        webgpu.renderContext.configure({ device, format: webgpu.presentationFormat });
        return webgpu;
    }
    createTsQuery(name, num) {
        if (this.device === null)
            throw 'no device';
        this.queries.set(name, this.device.createQuerySet({
            type: 'timestamp',
            count: num,
        }));
    }
    async shader(shaderfile) {
        if (this.device === null)
            throw 'no device';
        const shadertext = await fetch(shaderfile).then((r) => r.text());
        return this.device.createShaderModule({ code: shadertext });
    }
    createBuffer(name, size, usage) {
        if (this.device === null)
            throw 'no device';
        const old = this.buffers.get(name);
        if (old)
            old.destroy();
        this.buffers.set(name, this.device.createBuffer({
            size: size * 4,
            usage: usage,
        }));
        for (const pass of this.passes) {
            if ((pass.compute && pass.compute.bindings.has(name)) || (pass.render && pass.render.bindings.has(name))) {
                this.updateBindGroup(pass);
            }
        }
    }
    updateBindGroup(pass) {
        if (this.device === null)
            throw 'no device';
        const bindPass = pass.compute ? pass.compute : pass.render ? pass.render : null;
        if (bindPass) {
            const entries = new Array();
            let missing = false;
            for (const [name, location] of bindPass.bindings) {
                const buffer = this.buffers.get(name);
                if (buffer) {
                    entries.push({ binding: location, resource: { buffer: buffer } });
                }
                else {
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
    addComputePass(shader, entry, invocations, bindings, measure = null) {
        this.addComputePassInternal(shader, entry, invocations, bindings, measure);
    }
    addComputePassIndirect(shader, entry, indirectbuffer, bindings, measure = null) {
        this.addComputePassInternal(shader, entry, indirectbuffer, bindings, measure);
    }
    addComputePassInternal(shader, entry, invocations, bindings, measure) {
        if (this.device === null)
            throw 'no device';
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shader,
                entryPoint: entry,
            },
        });
        const passDescriptor = {};
        if (measure) {
            const query = this.queries.get(measure.query);
            if (query) {
                passDescriptor.timestampWrites = { querySet: query };
                if (measure.begin !== undefined) {
                    passDescriptor.timestampWrites.beginningOfPassWriteIndex = measure.begin;
                }
                if (measure.end !== undefined) {
                    passDescriptor.timestampWrites.endOfPassWriteIndex = measure.end;
                }
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
    addRenderPass(shader, vsentry, fsentry, vertices, bindings, measure = null) {
        if (this.device === null)
            throw 'no device';
        if (this.presentationFormat === null)
            throw 'no presentation format';
        if (this.renderContext === null)
            throw 'no render context';
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
        const passDescriptor = {
            colorAttachments: [
                {
                    loadOp: 'clear',
                    storeOp: 'store',
                    view: this.renderContext.getCurrentTexture().createView(),
                },
            ],
        };
        if (measure) {
            const query = this.queries.get(measure.query);
            if (query) {
                passDescriptor.timestampWrites = { querySet: query };
                if (measure.begin !== undefined) {
                    passDescriptor.timestampWrites.beginningOfPassWriteIndex = measure.begin;
                }
                if (measure.end !== undefined) {
                    passDescriptor.timestampWrites.endOfPassWriteIndex = measure.end;
                }
            }
        }
        const pass = {
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
    addBufferDownload(name, maxflight, callback) {
        const pass = {
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
    addClear(name) {
        const pass = {
            clear: {
                buffer: name,
            },
        };
        this.passes.push(pass);
    }
    addResolveQuery(queryName, bufferName) {
        const pass = {
            resolve: {
                query: queryName,
                buffer: bufferName,
            },
        };
        this.passes.push(pass);
    }
    updateBuffer(name, values) {
        if (this.device === null)
            throw 'no device';
        const buffer = this.buffers.get(name);
        if (buffer)
            this.device.queue.writeBuffer(buffer, 0, values);
    }
    execute() {
        if (this.device === null)
            throw 'no device';
        if (this.renderContext === null)
            throw 'no render context';
        const encoder = this.device.createCommandEncoder();
        for (const passDef of this.passes) {
            if (passDef.clear) {
                const buffer = this.buffers.get(passDef.clear.buffer);
                if (buffer) {
                    encoder.clearBuffer(buffer);
                }
            }
            else if (passDef.compute) {
                const pass = encoder.beginComputePass(passDef.compute.descriptor);
                pass.setPipeline(passDef.compute.pipeline);
                pass.setBindGroup(0, passDef.compute.bindGroup);
                if (typeof passDef.compute.invocations === 'number' && Number.isInteger(passDef.compute.invocations)) {
                    pass.dispatchWorkgroups(passDef.compute.invocations);
                }
                else if (typeof passDef.compute.invocations === 'string') {
                    const buffer = this.buffers.get(passDef.compute.invocations);
                    if (buffer)
                        pass.dispatchWorkgroupsIndirect(buffer, 0);
                }
                pass.end();
            }
            else if (passDef.render) {
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
            }
            else if (passDef.resolve) {
                const buffer = this.buffers.get(passDef.resolve.buffer);
                const query = this.queries.get(passDef.resolve.query);
                if (buffer && query) {
                    encoder.resolveQuerySet(query, 0, query.count, buffer, 0);
                }
            }
            else if (passDef.download) {
                passDef.download.stagingBuffer = null;
                const src = this.buffers.get(passDef.download.buffer);
                if (src) {
                    if (passDef.download.inFlight < passDef.download.maxFlight) {
                        const stagingBuffer = passDef.download.stagingQueue.pop() ||
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
    downloadInfo() {
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