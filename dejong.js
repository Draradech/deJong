import { RollingAverage, PointGraph, getCanvas, getInput, $ } from './utils.js';
import { WebGPU } from './webgpu.js';
async function main() {
    ///////////////////////////////
    // webgpu setup              //
    ///////////////////////////////
    const webgpu = await WebGPU.create(getCanvas('attractor'));
    const shader = await webgpu.shader('shader.wgsl');
    webgpu.createTsQuery('tsquery', 2);
    const uniformValues = new Float32Array(10);
    webgpu.createBuffer('uniform', uniformValues.length, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    webgpu.createBuffer('frameinfo', 13, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    webgpu.createBuffer('timestamp', 4, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.STORAGE);
    webgpu.createBuffer('indirect', 3, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE);
    function createDataBuffer(bufsize) {
        webgpu.createBuffer('data', bufsize + 1, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        // why +1?: on chrome-windows-nvidia4070 this shader runs more than twice as fast with buffers not a multiple of 4096
    }
    webgpu.addClear('data');
    let bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 4);
    webgpu.addComputePass(shader, 'deJong', 16, bindings, { query: 'tsquery', begin: 0, end: 1 });
    webgpu.addResolveQuery('tsquery', 'timestamp');
    bindings = new Map().set('uniform', 0).set('timestamp', 1).set('frameinfo', 2).set('indirect', 3);
    webgpu.addComputePass(shader, 'pass1t', 1, bindings);
    bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 4);
    webgpu.addComputePassIndirect(shader, 'deJong', 'indirect', bindings, { query: 'tsquery', begin: 0, end: 1 });
    webgpu.addResolveQuery('tsquery', 'timestamp');
    bindings = new Map().set('uniform', 0).set('timestamp', 1).set('frameinfo', 2).set('indirect', 3);
    webgpu.addComputePass(shader, 'pass2t', 1, bindings);
    bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 4);
    webgpu.addComputePassIndirect(shader, 'deJong', 'indirect', bindings, { query: 'tsquery', begin: 0, end: 1 });
    webgpu.addResolveQuery('tsquery', 'timestamp');
    bindings = new Map().set('timestamp', 1).set('frameinfo', 2);
    webgpu.addComputePass(shader, 'pass3t', 1, bindings);
    bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 5);
    webgpu.addRenderPass(shader, 'vs', 'fs', 3, bindings, { query: 'tsquery', begin: 0, end: 1 });
    bindings = new Map().set('timestamp', 1).set('frameinfo', 2);
    webgpu.addComputePass(shader, 'passrt', 1, bindings);
    webgpu.addBufferDownload('frameinfo', 4, readback);
    /////////////////////////////////
    // time info readback from gpu //
    /////////////////////////////////
    const gpuAverage = Array.from({ length: 5 }, () => new RollingAverage(60));
    const pointsAverage = Array.from({ length: 4 }, () => new RollingAverage(60));
    let gtime, npoints, p1time, p2time, p3time;
    function readback(arrayBuffer) {
        const frameinfo = new Uint32Array(arrayBuffer);
        const tsres = parseFloat(getInput('tsres').value);
        p1time = (((frameinfo[1] - frameinfo[0]) | 0) * tsres) / 1000000;
        p2time = (((frameinfo[3] - frameinfo[2]) | 0) * tsres) / 1000000;
        p3time = (((frameinfo[5] - frameinfo[4]) | 0) * tsres) / 1000000;
        gtime = (((frameinfo[7] - frameinfo[0]) | 0) * tsres) / 1000000;
        npoints = frameinfo[11];
        gpuAverage[0].addSample((((frameinfo[1] - frameinfo[0]) | 0) * tsres) / 1000000);
        gpuAverage[1].addSample((((frameinfo[3] - frameinfo[2]) | 0) * tsres) / 1000000);
        gpuAverage[2].addSample((((frameinfo[5] - frameinfo[4]) | 0) * tsres) / 1000000);
        gpuAverage[3].addSample((((frameinfo[7] - frameinfo[6]) | 0) * tsres) / 1000000);
        gpuAverage[4].addSample((((frameinfo[7] - frameinfo[0]) | 0) * tsres) / 1000000);
        pointsAverage[0].addSample(frameinfo[8]);
        pointsAverage[1].addSample(frameinfo[9]);
        pointsAverage[2].addSample(frameinfo[10]);
        pointsAverage[3].addSample(frameinfo[11]);
    }
    ///////////////////////////////
    // ui setup                  //
    ///////////////////////////////
    let canvasDeviceWidth, canvasDeviceHeight;
    let firstrender = true;
    const canvas = getCanvas('attractor');
    function resize() {
        const scale = 0.01 * parseFloat(getInput('scale').value);
        const width = canvasDeviceWidth * scale;
        const height = canvasDeviceHeight * scale;
        const limit = 8192;
        canvas.width = Math.max(1, Math.min(width, limit));
        canvas.height = Math.max(1, Math.min(height, limit));
        createDataBuffer(canvas.width * canvas.height * 3);
        if (firstrender) {
            requestAnimationFrame(render);
            firstrender = false;
        }
    }
    const observer = new ResizeObserver((e) => {
        canvasDeviceWidth =
            e[0].devicePixelContentBoxSize?.[0].inlineSize || e[0].contentBoxSize?.[0].inlineSize * devicePixelRatio;
        canvasDeviceHeight =
            e[0].devicePixelContentBoxSize?.[0].blockSize || e[0].contentBoxSize?.[0].blockSize * devicePixelRatio;
        resize();
    });
    observer.observe(canvas);
    getInput('scale').onchange = resize;
    function toggleUi() {
        if (($('graph').style.visibility || 'visible') == 'visible') {
            $('graph').style.visibility = 'hidden';
            $('info').style.visibility = 'hidden';
            $('input').style.visibility = 'hidden';
        }
        else if (($('what').style.visibility || 'visible') == 'visible') {
            $('what').style.visibility = 'hidden';
        }
        else {
            $('graph').style.visibility = 'visible';
            $('info').style.visibility = 'visible';
            $('what').style.visibility = 'visible';
            $('input').style.visibility = 'visible';
        }
    }
    document.onkeydown = (evt) => {
        evt = evt || window.event;
        if (evt.key == 'h') {
            toggleUi();
        }
    };
    canvas.onclick = () => {
        if (document.fullscreenElement != null) {
            toggleUi();
        }
        else {
            document.documentElement.requestFullscreen();
        }
    };
    ////////////////////////////////
    // frame (render + ui update) //
    ////////////////////////////////
    const frameAverage = new RollingAverage(60);
    const jsAverage = new RollingAverage(60);
    function updateInfoPanel(downloadInfo) {
        $('info').textContent = `\
frame:   ${frameAverage.get().toFixed(2)}ms\
 (${(1000 / frameAverage.get()).toFixed(1)}fps)
js:      ${jsAverage.get().toFixed(2)}ms

pass 1:  ${gpuAverage[0].get().toFixed(2)}ms\
 ${(pointsAverage[0].get() / 1e6).toFixed(1)}M
pass 2:  ${gpuAverage[1].get().toFixed(2)}ms\
 ${(pointsAverage[1].get() / 1e6).toFixed(1)}M
pass 3:  ${gpuAverage[2].get().toFixed(2)}ms\
 ${(pointsAverage[2].get() / 1e6).toFixed(1)}M
render:  ${gpuAverage[3].get().toFixed(2)}ms
total:   ${gpuAverage[4].get().toFixed(2)}ms\
 ${(pointsAverage[3].get() / 1e6).toFixed(1)}M

txsize:  ${canvas.width}x${canvas.height}\
 (${((canvas.width * canvas.height * 4 * 3) / 1024 / 1024).toFixed(1)}MB)

queue:   ${downloadInfo.staging}
flight:  ${downloadInfo.flight}`;
    }
    let t = Math.random() * 1e3;
    const frameGraph = new PointGraph(getCanvas('graph'));
    let ftime, jtime;
    let frame = 0;
    let then = 0;
    function render(now) {
        // timing
        const startTime = performance.now();
        ftime = now - then;
        then = now;
        frameAverage.addSample(ftime);
        // update uniforms
        frame++;
        if (getInput('animate_t').checked) {
            t += 1e-6 * parseInt(getInput('speed').value);
            getInput('value_t').value = t.toFixed(3);
        }
        else {
            t = parseFloat(getInput('value_t').value);
        }
        if (getInput('animate_a').checked) {
            uniformValues[0] = 4 * Math.sin(t * 1.03);
            getInput('value_a').value = uniformValues[0].toFixed(3);
        }
        else {
            uniformValues[0] = parseFloat(getInput('value_a').value);
        }
        if (getInput('animate_b').checked) {
            uniformValues[1] = 4 * Math.sin(t * 1.07);
            getInput('value_b').value = uniformValues[1].toFixed(3);
        }
        else {
            uniformValues[1] = parseFloat(getInput('value_b').value);
        }
        if (getInput('animate_c').checked) {
            uniformValues[2] = 4 * Math.sin(t * 1.09);
            getInput('value_c').value = uniformValues[2].toFixed(3);
        }
        else {
            uniformValues[2] = parseFloat(getInput('value_c').value);
        }
        if (getInput('animate_d').checked) {
            uniformValues[3] = 4 * Math.sin(t * 1.13);
            getInput('value_d').value = uniformValues[3].toFixed(3);
        }
        else {
            uniformValues[3] = parseFloat(getInput('value_d').value);
        }
        uniformValues[4] = frame;
        uniformValues[5] = canvas.width;
        uniformValues[6] = parseInt(getInput('loop').value) || 32;
        uniformValues[7] = parseFloat(getInput('bright').value) * 4.9e-6;
        uniformValues[8] = Math.max(Math.min(parseFloat(getInput('budget').value) || 12, 100), 0.01);
        uniformValues[9] = parseFloat(getInput('tsres').value);
        // execute
        webgpu.updateBuffer('uniform', uniformValues);
        webgpu.execute();
        // ui
        updateInfoPanel(webgpu.downloadInfo());
        frameGraph.begin();
        frameGraph.drawPoint(gtime, 'red', -1, 35);
        frameGraph.drawPoint(p1time, 'darkgreen', -1, 35);
        frameGraph.drawPoint(p2time, 'darkgoldenrod', -1, 35);
        frameGraph.drawPoint(p3time, 'darkred', -1, 35);
        frameGraph.drawPoint(ftime, 'lime', -1, 35);
        frameGraph.drawPoint(jtime, 'yellow', -1, 35);
        frameGraph.drawPoint(npoints, 'cyan', 0.1e6, 1e9, true);
        frameGraph.end();
        // more timing
        jtime = performance.now() - startTime;
        jsAverage.addSample(jtime);
        // next
        requestAnimationFrame(render);
    }
}
main();
