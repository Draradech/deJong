# Peter de Jong attractor
## Equation:
```
x2 = sin(a * y1) - cos(b * x1)
y2 = sin(c * x1) - cos(d * y1)
```
## Coloured by:
```
red:   abs(x2 - x1)
green: abs(y2 - y1)
blue:  1.0
```
## Animated by:
```
a = 4 * sin(t * 1.03)
b = 4 * sin(t * 1.07)
c = 4 * sin(t * 1.09)
d = 4 * sin(t * 1.13)
```
## WebGPU
If your browser supports WebGPU, you can try it here: https://draradech.github.io/deJong/webgpu/
