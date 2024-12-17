export class RollingAverage {
  private total = 0;
  private samples: Array<number> = [];
  private cursor = 0;
  private numSamples;

  constructor(numSamples = 30) {
    this.numSamples = numSamples;
  }

  public addSample(v: number) {
    if (Number.isFinite(v)) {
      const v2 = Math.max(v, 0);
      this.total += v2 - (this.samples[this.cursor] || 0);
      this.samples[this.cursor] = v2;
      this.cursor = (this.cursor + 1) % this.numSamples;
    }
  }

  public get() {
    return this.total / this.samples.length;
  }
}

export class PointGraph {
  private context;
  private step;
  constructor(canvas: HTMLCanvasElement, step = 2) {
    this.context = canvas.getContext('2d');
    this.step = step;
  }
  public begin() {
    if (this.context) {
      this.context.globalCompositeOperation = 'copy';
      this.context.drawImage(this.context.canvas, -this.step, 0);
      this.context.globalCompositeOperation = 'source-over';
    }
  }
  public drawPoint(value: number, style: string, min: number, max: number, log = false) {
    if (this.context) {
      this.context.fillStyle = style;
      const w = this.context.canvas.width;
      const h = this.context.canvas.height;
      if (log) {
        value = Math.log10(value / min) / Math.log10(max / min);
      } else {
        value = (value - min) / (max - min);
      }
      const y = h * (1 - value);
      this.context.fillRect(w - this.step, y, this.step, this.step);
    }
  }
  public end() {}
}

export function $(id: string) {
  const e = document.getElementById(id);
  if (!(e instanceof HTMLElement)) {
    throw `Couldn't find element "${id}"`;
  } else {
    return e;
  }
}

export function getCanvas(id: string) {
  const e = $(id);
  if (!(e instanceof HTMLCanvasElement)) {
    throw `Couldn't find canvas "${id}"`;
  } else {
    return e;
  }
}

export function getInput(id: string) {
  const e = $(id);
  if (!(e instanceof HTMLInputElement)) {
    throw `Couldn't find input element "${id}"`;
  } else {
    return e;
  }
}
