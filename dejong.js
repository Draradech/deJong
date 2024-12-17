import { RollingAverage, PointGraph, getCanvas, getInput, $ } from './utils.js';
import { WebGPU } from './webgpu.js';
async function main() {
    ///////////////////////////////
    // webgpu setup              //
    ///////////////////////////////
    const webgpu = await WebGPU.create(getCanvas('attractor'), 2);
    const shader = await webgpu.shader('shader.wgsl');
    const uniformValues = new Float32Array(9);
    webgpu.createBuffer('uniform', uniformValues.length, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    webgpu.createBuffer('frameinfo', 13, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    webgpu.createBuffer('timestamp', 4, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.STORAGE);
    webgpu.createBuffer('indirect', 3, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE);
    function createDataBuffer(bufsize) {
        webgpu.createBuffer('data', bufsize + 1, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    }
    let bindings;
    bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 4);
    webgpu.addComputePass(shader, 'deJong', 16, bindings, { begin: 0, end: 1 });
    bindings = new Map().set('uniform', 0).set('timestamp', 1).set('frameinfo', 2).set('indirect', 3);
    webgpu.addComputePass(shader, 'pass2', 1, bindings);
    bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 4);
    webgpu.addComputePass(shader, 'deJong', 'indirect', bindings, { begin: 0, end: 1 });
    bindings = new Map().set('uniform', 0).set('timestamp', 1).set('frameinfo', 2).set('indirect', 3);
    webgpu.addComputePass(shader, 'pass4', 1, bindings);
    bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 4);
    webgpu.addComputePass(shader, 'deJong', 'indirect', bindings, { begin: 0, end: 1 });
    bindings = new Map().set('timestamp', 1).set('frameinfo', 2);
    webgpu.addComputePass(shader, 'pass6', 1, bindings);
    bindings = new Map().set('uniform', 0).set('frameinfo', 2).set('data', 5);
    webgpu.addRenderPass(shader, 'vs', 'fs', 3, bindings, { begin: 0, end: 1 });
    bindings = new Map().set('timestamp', 1).set('frameinfo', 2);
    webgpu.addComputePass(shader, 'pass7', 1, bindings);
    /////////////////////////////////
    // time info readback from gpu //
    /////////////////////////////////
    const gpuAverage = Array.from({ length: 5 }, () => new RollingAverage(60));
    const pointsAverage = Array.from({ length: 4 }, () => new RollingAverage(60));
    let gtime, npoints;
    function readback(arrayBuffer) {
        const frameinfo = new Uint32Array(arrayBuffer);
        gtime = (frameinfo[7] - frameinfo[0]) / 1000000;
        npoints = frameinfo[11];
        gpuAverage[0].addSample(((frameinfo[1] - frameinfo[0]) | 0) / 1000000);
        gpuAverage[1].addSample(((frameinfo[3] - frameinfo[2]) | 0) / 1000000);
        gpuAverage[2].addSample(((frameinfo[5] - frameinfo[4]) | 0) / 1000000);
        gpuAverage[3].addSample(((frameinfo[7] - frameinfo[6]) | 0) / 1000000);
        gpuAverage[4].addSample(((frameinfo[7] - frameinfo[0]) | 0) / 1000000);
        pointsAverage[0].addSample(frameinfo[8]);
        pointsAverage[1].addSample(frameinfo[9]);
        pointsAverage[2].addSample(frameinfo[10]);
        pointsAverage[3].addSample(frameinfo[11]);
    }
    webgpu.addBufferDownload('frameinfo', 4, readback);
    ///////////////////////////////
    // ui setup                  //
    ///////////////////////////////
    let cw, cy;
    let firstrender = true;
    const canvas = getCanvas('attractor');
    function resize() {
        const scale = 0.01 * parseFloat(getInput('scale').value);
        const width = cw * scale;
        const height = cy * scale;
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
        cw = e[0].devicePixelContentBoxSize?.[0].inlineSize || e[0].contentBoxSize?.[0].inlineSize * devicePixelRatio;
        cy = e[0].devicePixelContentBoxSize?.[0].blockSize || e[0].contentBoxSize?.[0].blockSize * devicePixelRatio;
        resize();
    });
    observer.observe(canvas);
    getInput('scale').onchange = resize;
    function updateInput() {
        uniformValues[6] = parseInt(getInput('loop').value);
        uniformValues[7] = parseFloat(getInput('bright').value) * 4.9e-6;
        uniformValues[8] = parseFloat(getInput('budget').value);
    }
    getInput('loop').onchange = updateInput;
    getInput('bright').onchange = updateInput;
    getInput('budget').onchange = updateInput;
    updateInput();
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
    const frameGraph = new PointGraph(getCanvas('graph'));
    let ftime, jtime;
    let frame = 0;
    let t = Math.random() * 1e3;
    let then = 0;
    function updateInfoPanel(downloadInfo) {
        $('info').textContent = `\
frame:   ${frameAverage.get().toFixed(2)}ms\
 (${(1000 / frameAverage.get()).toFixed(1)}fps)
js:      ${jsAverage.get().toFixed(2)}ms

pass1:   ${gpuAverage[0].get().toFixed(2)}ms\
 ${(pointsAverage[0].get() / 1e6).toFixed(1)}M
pass3:   ${gpuAverage[1].get().toFixed(2)}ms\
 ${(pointsAverage[1].get() / 1e6).toFixed(1)}M
pass5:   ${gpuAverage[2].get().toFixed(2)}ms\
 ${(pointsAverage[2].get() / 1e6).toFixed(1)}M
render:  ${gpuAverage[3].get().toFixed(2)}ms
total:   ${gpuAverage[4].get().toFixed(2)}ms\
 ${(pointsAverage[3].get() / 1e6).toFixed(1)}M

txsize:  ${canvas.width}x${canvas.height}\
 (${((canvas.width * canvas.height * 4 * 3) / 1024 / 1024).toFixed(1)}MB)
queue:   ${downloadInfo.staging}
flight:  ${downloadInfo.flight}

t:       ${t.toFixed(3)}
a:       ${uniformValues[0].toFixed(3)}
b:       ${uniformValues[1].toFixed(3)}
c:       ${uniformValues[2].toFixed(3)}
d:       ${uniformValues[3].toFixed(3)}

press 'h' to hide ui`;
    }
    function render(now) {
        // timing
        const startTime = performance.now();
        ftime = now - then;
        then = now;
        frameAverage.addSample(ftime);
        // update uniforms
        frame++;
        t += 1e-4;
        uniformValues[0] = 4 * Math.sin(t * 1.03);
        uniformValues[1] = 4 * Math.sin(t * 1.07);
        uniformValues[2] = 4 * Math.sin(t * 1.09);
        uniformValues[3] = 4 * Math.sin(t * 1.13);
        uniformValues[4] = frame;
        uniformValues[5] = canvas.width;
        // execute
        webgpu.updateBuffer('uniform', uniformValues);
        webgpu.execute();
        // ui
        updateInfoPanel(webgpu.downloadInfo());
        frameGraph.begin();
        frameGraph.drawPoint(gtime, 'red', 0, 35);
        frameGraph.drawPoint(ftime, 'lime', 0, 35);
        frameGraph.drawPoint(jtime, 'yellow', 0, 35);
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
