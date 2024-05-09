// pub fn polyline(
//     points: Vec<Point3<f32>>,
//     normals: Vec<Vector3<f32>>,
//     width: Vec<f32>,
//     colors: Vec<[f32; 3]>,
// ) -> Vec<Vertex3D> {
//     assert!(points.len() > 1, "not enough points");
//     assert!(
//         points.len() == normals.len(),
//         "there must be exactly one normal per point"
//     );
//     assert!(
//         points.len() == width.len(),
//         "there must be exactly one width per point"
//     );
//     assert!(
//         points.len() - 1 == colors.len(),
//         "there must be exactly one color per line segment"
//     );
//     // find the vector of each line segment
//     let dposition_per_segment: Vec<Vector3<f32>> = points.windows(2).map(|w| w[1] - w[0]).collect();

//     // dposition_per_points[0] = dposition_per_segment[0] and dposition_per_points[n] = dposition_per_segment[n-1], but it is the average of the two for the points in between
//     let dposition_per_points: Vec<Vector3<f32>> = {
//         let mut dposition_per_points = Vec::new();
//         dposition_per_points.push(dposition_per_segment[0]);
//         for i in 1..dposition_per_segment.len() {
//             dposition_per_points
//                 .push((dposition_per_segment[i - 1] + dposition_per_segment[i]).normalize());
//         }
//         dposition_per_points.push(dposition_per_segment[dposition_per_segment.len() - 1]);
//         dposition_per_points
//     };

//     // find the cross vectors (along which the width will be applied)
//     let cross_vectors: Vec<Vector3<f32>> = dposition_per_points
//         .iter()
//         .zip(normals.iter())
//         .map(|(&v, n)| v.cross(n).normalize())
//         .collect();

//     // find the left and right points
//     let left_points: Vec<Point3<f32>> = cross_vectors
//         .iter()
//         .zip(width.iter())
//         .zip(points.iter())
//         .map(|((v, &w), p)| p - v * w)
//         .collect();

//     let right_points: Vec<Point3<f32>> = cross_vectors
//         .iter()
//         .zip(width.iter())
//         .zip(points.iter())
//         .map(|((v, &w), p)| p + v * w)
//         .collect();

//     let vertexes: Vec<Vertex3D> = std::iter::zip(left_points.windows(2), right_points.windows(2))
//         .zip(colors)
//         .flat_map(|((l, r), color)| {
//             vec![
//                 Vertex3D::new(r[0].into(), color),
//                 Vertex3D::new(l[1].into(), color),
//                 Vertex3D::new(l[0].into(), color),
//                 Vertex3D::new(r[1].into(), color),
//                 Vertex3D::new(l[1].into(), color),
//                 Vertex3D::new(r[0].into(), color),
//             ]
//         })
//         .collect();
//     vertexes
// }

import {vec3} from 'gl-matrix';
import { Vertex } from './vertex';

export function polyline(
    points: vec3[],
    normals: vec3[],
    width: number[],
    colors: vec3[],
): Vertex[] {
    if (points.length <= 1) {
        throw new Error("not enough points");
    }
    if (points.length !== normals.length) {
        throw new Error("there must be exactly one normal per point");
    }
    if (points.length !== width.length) {
        throw new Error("there must be exactly one width per point");
    }
    if (points.length - 1 !== colors.length) {
        throw new Error("there must be exactly one color per line segment");
    }
    // find the vector of each line segment
    let dposition_per_segment: vec3[] = [];
    for(let i = 0; i < points.length-1; i++) {
        dposition_per_segment.push(vec3.subtract(vec3.create(), points[i + 1], points[i]));
    }

    // dposition_per_points[0] = dposition_per_segment[0] and dposition_per_points[n] = dposition_per_segment[n-1], but it is the average of the two for the points in between
    let dposition_per_points: vec3[] = [];
    dposition_per_points.push(dposition_per_segment[0]);
    for (let i = 1; i < dposition_per_segment.length; i++) {
        dposition_per_points.push(vec3.scale(vec3.create(), vec3.add(vec3.create(), dposition_per_segment[i - 1], dposition_per_segment[i]), 0.5));
    }
    dposition_per_points.push(dposition_per_segment[dposition_per_segment.length - 1]);

    // find the cross vectors (along which the width will be applied)
    let cross_vectors: vec3[] = dposition_per_points.map((v, i) => vec3.normalize(vec3.create(), vec3.cross(vec3.create(), v, normals[i])));

    // find the left and right points
    let left_points: vec3[] = cross_vectors.map((v, i) => vec3.subtract(vec3.create(), points[i], vec3.scale(vec3.create(), v, width[i])));

    let right_points: vec3[] = cross_vectors.map((v, i) => vec3.add(vec3.create(), points[i], vec3.scale(vec3.create(), v, width[i])));

    let vertexes: Vertex[] = [];
    for(let i = 0; i < left_points.length-1; i++) {
        vertexes.push({
            position: right_points[i],
            color: colors[i],
        });
        vertexes.push({
            position: left_points[i + 1],
            color: colors[i],
        });
        vertexes.push({
            position: left_points[i],
            color: colors[i],
        });
        vertexes.push({
            position: right_points[i + 1],
            color: colors[i],
        });
        vertexes.push({
            position: left_points[i + 1],
            color: colors[i],
        });
        vertexes.push({
            position: right_points[i],
            color: colors[i],
        });
    }
    console.log(vertexes)
    return vertexes;
}