import { vec3 } from "gl-matrix";
import { Vertex } from "./vertex";


export function axesLines(scale: number = 1) {
    const vertices: Vertex[] = [];
    const colors = [
        vec3.fromValues(1, 0, 0),
        vec3.fromValues(0, 1, 0),
        vec3.fromValues(0, 0, 1),
    ];
    const directions = [
        vec3.fromValues(scale, 0, 0),
        vec3.fromValues(0, scale, 0),
        vec3.fromValues(0, 0, scale),
    ];
    for (let i = 0; i < 3; i++) {
        vertices.push({ position: vec3.create(), color: colors[i] });
        vertices.push({ position: directions[i], color: colors[i] });
    }
    return vertices;
}