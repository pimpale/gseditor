import { mat3, quat, vec3, vec4 } from 'gl-matrix';

// a raw input object
export type GaussianObjectInput = {
    count: number,
    // gaussian positions, size 3 * gaussianCount
    positions: Float32Array,
    // gaussian opacities, size gaussianCount
    opacities: Float32Array,
    // gaussian colors, size 3 * gaussianCount
    colors: Float32Array,
    // gaussian rotation quaternions, size 4 * gaussianCount
    rotations: Float32Array
    // gaussian scales, size 3 * gaussianCount
    scales: Float32Array
}

// Load all gaussian data from a point-cloud file
// Original C++ implementation: https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_union/src/projects/gaussianviewer/renderer/GaussianView.cpp#L70
export function loadPly(content: ArrayBuffer): GaussianObjectInput {
    // Read header
    const start = performance.now()
    const contentStart = new TextDecoder('utf-8').decode(content.slice(0, 2000))
    const headerEnd = contentStart.indexOf('end_header') + 'end_header'.length + 1
    const [header] = contentStart.split('end_header')

    // Get number of gaussians
    const regex = /element vertex (\d+)/;
    const match = header.match(regex);
    if (!match) {
        throw new Error('Invalid PLY file: missing "element vertex"');
    }
    const gaussianCount = parseInt(match[1]);

    // Create arrays for gaussian properties
    const positions: number[] = []
    const opacities: number[] = []
    const rotations: number[] = []
    const scales: number[] = []
    const harmonics: number[] = []
    const colors: number[] = []

    // Helpers
    const sigmoid = (m1: number) => 1 / (1 + Math.exp(-m1))
    const NUM_PROPS = 62

    // Create a dataview to access the buffer's content on a byte levele
    const view = new DataView(content)

    // Get a slice of the dataview relative to a splat index
    const fromDataViewArr = (splatID: number, start: number, end: number) => {
        const startOffset = headerEnd + splatID * NUM_PROPS * 4 + start * 4

        return new Float32Array(end - start).map((_, i) => view.getFloat32(startOffset + i * 4, true))
    }

    const fromDataViewFloat = (splatID: number, start: number) => {
        const startOffset = headerEnd + splatID * NUM_PROPS * 4 + start * 4

        return view.getFloat32(startOffset, true)
    }

    // Extract all properties for a gaussian splat using the dataview
    const extractSplatData = (splatID: number) => {
        const position = fromDataViewArr(splatID, 0, 3)
        // const n = fromDataView(splatID, 3, 6) // Not used
        const harmonic = fromDataViewArr(splatID, 6, 9)

        const H_END = 6 + 48 // Offset of the last harmonic coefficient
        const opacity = fromDataViewFloat(splatID, H_END)
        const scale = fromDataViewArr(splatID, H_END + 1, H_END + 4)
        const rotation = fromDataViewArr(splatID, H_END + 4, H_END + 8)

        return { position, harmonic, opacity, scale, rotation }
    }

    for (let i = 0; i < gaussianCount; i++) {
        // Extract data for current gaussian
        let { position, harmonic, opacity, scale, rotation } = extractSplatData(i)

        // Normalize quaternion
        let length2 = 0

        for (let j = 0; j < 4; j++)
            length2 += rotation[j] * rotation[j]

        const length = Math.sqrt(length2)

        rotation = rotation.map(v => v / length)
        rotations.push(rotation[1], rotation[2], rotation[3], rotation[0])

        // Exponentiate scale
        scale = scale.map(v => Math.exp(v))
        scales.push(...scale)

        // Activate alpha
        opacity = sigmoid(opacity)
        opacities.push(opacity)

        // (Webgl-specific) Equivalent to computeColorFromSH() with degree 0:
        // Use the first spherical harmonic to pre-compute the color.
        // This allow to avoid sending harmonics to the web worker or GPU,
        // but removes view-dependent lighting effects like reflections.
        // If we were to use a degree > 0, we would need to recompute the color 
        // each time the camera moves, and send many more harmonics to the worker:
        // Degree 1: 4 harmonics needed (12 floats) per gaussian
        // Degree 2: 9 harmonics needed (27 floats) per gaussian
        // Degree 3: 16 harmonics needed (48 floats) per gaussian
        const SH_C0 = 0.28209479177387814
        const color = [
            0.5 + SH_C0 * harmonic[0],
            0.5 + SH_C0 * harmonic[1],
            0.5 + SH_C0 * harmonic[2]
        ]
        colors.push(...color)
        // harmonics.push(...harmonic)

        positions.push(...position)
    }

    console.log(`Loaded ${gaussianCount} gaussians in ${((performance.now() - start) / 1000).toFixed(3)}s`)
    
    return {
        count: gaussianCount,
        positions: Float32Array.from(positions),
        opacities: Float32Array.from(opacities),
        colors: Float32Array.from(colors),
        scales: Float32Array.from(scales),
        rotations: Float32Array.from(rotations)
    }
}
