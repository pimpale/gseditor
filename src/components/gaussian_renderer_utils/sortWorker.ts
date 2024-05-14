import { mat3, mat4, quat, vec3, vec4 } from "gl-matrix"
import assert from "../../utils/assert"
import { GaussianObjectInput } from "./sceneLoader"

type SortWorkerInput = {
    viewMatrix: Float32Array
    sortingAlgorithm: string
    sceneGraph: Map<number, { translate: vec3, rotate: quat, object: GaussianObjectInput }>
}

// a processed input object with object ids
export type ProcessedGaussianScene = {
    count: number,
    sceneMin: Float32Array,
    sceneMax: Float32Array,
    positions: Float32Array,
    opacities: Float32Array,
    colors: Float32Array,
    cov3Da: Float32Array,
    cov3Db: Float32Array,
    objectIds: Uint32Array
}

onmessage = (event: MessageEvent<SortWorkerInput>) => {

    console.log("[Worker] Received message")

    const { sceneGraph, viewMatrix, sortingAlgorithm } = event.data

    const start = performance.now()

    const count = [...sceneGraph.values()].reduce((a, b) => a + b.object.count, 0)

    // create a merged object for fast lookup
    const gaussians: ProcessedGaussianScene = {
        count,
        sceneMin: Float32Array.from([Infinity, Infinity, Infinity]),
        sceneMax: Float32Array.from([-Infinity, -Infinity, -Infinity]),
        colors: new Float32Array(count * 3),
        positions: new Float32Array(count * 3),
        opacities: new Float32Array(count),
        cov3Da: new Float32Array(count * 3),
        cov3Db: new Float32Array(count * 3),
        objectIds: new Uint32Array(count),
    }

    let offset = 0;
    for (const [key, value] of sceneGraph.entries()) {
        const id = key
        const g = value.object;
        gaussians.colors.set(g.colors, offset * 3)
        gaussians.opacities.set(g.opacities, offset)

        const g_positions = new Float32Array(g.count * 3);
        for (let i = 0; i < g.count; i++) {
            const pos = vec3.transformQuat(vec3.create(), g.positions.slice(i * 3, i * 3 + 3), value.rotate)
            vec3.add(pos, pos, value.translate)
            g_positions.set(pos, i * 3)
            gaussians.sceneMin = gaussians.sceneMin.map((v, j) => Math.min(v, pos[j]))
            gaussians.sceneMax = gaussians.sceneMax.map((v, j) => Math.max(v, pos[j]))
        }
        gaussians.positions.set(g_positions, offset * 3)

        const g_cov3Da = new Float32Array(g.count * 3)
        const g_cov3Db = new Float32Array(g.count * 3)
        for (let i = 0; i < g.count; i++) {
            const rotation = quat.multiply(quat.create(), g.rotations.slice(i * 4, i * 4 + 4), value.rotate);
            const [cov3Da, cov3Db] = computeCov3D(g.scales.slice(i * 3, i * 3 + 3), 1, rotation);
            g_cov3Da.set(cov3Da, i * 3)
            g_cov3Db.set(cov3Db, i * 3)
        }

        gaussians.cov3Da.set(g_cov3Da, offset * 3)
        gaussians.cov3Db.set(g_cov3Db, offset * 3)
        gaussians.objectIds.fill(id, offset, offset + g.count)
        offset += g.count
    }

    // create an empty object to write to
    const data: ProcessedGaussianScene = {
        count,
        sceneMin: gaussians.sceneMin,
        sceneMax: gaussians.sceneMax,
        colors: new Float32Array(count * 3),
        positions: new Float32Array(count * 3),
        opacities: new Float32Array(count),
        cov3Da: new Float32Array(count * 3),
        cov3Db: new Float32Array(count * 3),
        objectIds: new Uint32Array(count),
    }

    // Sort the gaussians!
    const depthIndex = sortGaussiansByDepth(gaussians.positions, viewMatrix, sortingAlgorithm)

    // Update arrays containing the data
    for (let j = 0; j < depthIndex.length; j++) {
        const i = depthIndex[j]

        data.colors[j * 3] = gaussians.colors[i * 3]
        data.colors[j * 3 + 1] = gaussians.colors[i * 3 + 1]
        data.colors[j * 3 + 2] = gaussians.colors[i * 3 + 2]

        data.positions[j * 3] = gaussians.positions[i * 3]
        data.positions[j * 3 + 1] = gaussians.positions[i * 3 + 1]
        data.positions[j * 3 + 2] = gaussians.positions[i * 3 + 2]

        data.opacities[j] = gaussians.opacities[i]
        data.objectIds[j] = gaussians.objectIds[i]

        // Split the covariance matrix into two vec3
        // so they can be used as vertex shader attributes
        data.cov3Da[j * 3] = gaussians.cov3Da[i * 3]
        data.cov3Da[j * 3 + 1] = gaussians.cov3Da[i * 3 + 1]
        data.cov3Da[j * 3 + 2] = gaussians.cov3Da[i * 3 + 2]

        data.cov3Db[j * 3] = gaussians.cov3Db[i * 3]
        data.cov3Db[j * 3 + 1] = gaussians.cov3Db[i * 3 + 1]
        data.cov3Db[j * 3 + 2] = gaussians.cov3Db[i * 3 + 2]
    }

    // Check that the arrays are the correct length
    assert(data.cov3Da.length == 3 * gaussians.count, "cov3Da count mismatch")
    assert(data.cov3Db.length == 3 * gaussians.count, "cov3Db count mismatch")

    const sortTime = `${((performance.now() - start) / 1000).toFixed(3)}s`
    console.log(`[Worker] Sorted ${gaussians.count} gaussians in ${sortTime}. Algorithm: ${sortingAlgorithm}`)

    postMessage({
        data, sortTime,
    }, [
        data.positions.buffer,
        data.colors.buffer,
        data.opacities.buffer,
        data.cov3Da.buffer,
        data.cov3Db.buffer,
        data.objectIds.buffer,
    ])

}



function sortGaussiansByDepth(gaussian_positions: Float32Array, viewMatrix: Float32Array, sortingAlgorithm: string): Uint32Array {
    const n_gaussians = gaussian_positions.length / 3

    const calcDepth = (i: number) => gaussian_positions[i * 3] * viewMatrix[2] +
        gaussian_positions[i * 3 + 1] * viewMatrix[6] +
        gaussian_positions[i * 3 + 2] * viewMatrix[10]

    const depthIndex = new Uint32Array(n_gaussians)

    // Default javascript sort [~0.9s]
    if (sortingAlgorithm == 'Array.sort') {
        const indices = new Array(n_gaussians)
            .fill(0)
            .map((_, i) => ({
                depth: calcDepth(i),
                index: i
            }))
            .sort((a, b) => a.depth - b.depth)
            .map(v => v.index)

        depthIndex.set(indices)
    }
    // Quick sort algorithm (Hoare partition scheme) [~0.4s]
    else if (sortingAlgorithm == 'quick sort') {
        const depths = new Float32Array(n_gaussians)

        for (let i = 0; i < n_gaussians; i++) {
            depthIndex[i] = i
            depths[i] = calcDepth(i)
        }

        quicksort(depths, depthIndex, 0, n_gaussians - 1)
    }
    // 16 bit single-pass counting sort [~0.3s]
    // https://github.com/antimatter15/splat
    else if (sortingAlgorithm == 'count sort') {
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(n_gaussians);

        for (let i = 0; i < n_gaussians; i++) {
            const depth = (calcDepth(i) * 4096) | 0

            sizeList[i] = depth
            maxDepth = Math.max(maxDepth, depth)
            minDepth = Math.min(minDepth, depth)
        }

        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < n_gaussians; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
        for (let i = 0; i < n_gaussians; i++) depthIndex[starts0[sizeList[i]]++] = i;
    }

    return depthIndex;
}

// Quicksort algorithm - https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
function quicksort(A: Float32Array, B: Float32Array | Uint32Array, lo: number, hi: number) {
    if (lo < hi) {
        const p = partition(A, B, lo, hi)
        quicksort(A, B, lo, p)
        quicksort(A, B, p + 1, hi)
    }
}

function partition(A: Float32Array, B: Float32Array | Uint32Array, lo: number, hi: number) {
    const pivot = A[Math.floor((hi - lo) / 2) + lo]
    let i = lo - 1
    let j = hi + 1

    while (true) {
        do { i++ } while (A[i] < pivot)
        do { j-- } while (A[j] > pivot)

        if (i >= j) return j

        let tmp = A[i]; A[i] = A[j]; A[j] = tmp // Swap A
        tmp = B[i]; B[i] = B[j]; B[j] = tmp // Swap B
    }
}


// Converts scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space.
// Original CUDA implementation: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L118
function computeCov3D(scale: vec3, mod: number, rot: vec4): [number[], number[]] {
    const tmp = mat3.create()
    const S = mat3.create()
    const R = mat3.create()
    const M = mat3.create()
    const Sigma = mat3.create()

    // Create scaling matrix
    mat3.set(S,
        mod * scale[0], 0, 0,
        0, mod * scale[1], 0,
        0, 0, mod * scale[2]
    )

    const r = rot[0]
    const x = rot[1]
    const y = rot[2]
    const z = rot[3]

    // Compute rotation matrix from quaternion
    mat3.set(R,
        1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
        2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
        2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y)
    )

    mat3.multiply(M, S, R)  // M = S * R

    // Compute 3D world covariance matrix Sigma
    mat3.multiply(Sigma, mat3.transpose(tmp, M), M)  // Sigma = transpose(M) * M

    // Covariance is symmetric, only store upper right
    return [
        [Sigma[0], Sigma[1], Sigma[2]],
        [Sigma[4], Sigma[5], Sigma[8]]
    ]
}