struct VertexOutput {
	[[builtin(position)]] clip_position: vec4<f32>;
	// p here is a duplicate of clip_position, as clip_position
	// doesn't interpolate.
	[[location(0)]] p: vec2<f32>;
};

[[block]]
struct DataUniform {
	cr: f32;
	ci: f32;
	// treat as boolean
	mandel: u32;
	depth: u32;
	sclex: f32;
	scley: f32;
	trslx: f32;
	trsly: f32;
};
[[group(0), binding(0)]]
var<uniform> data: DataUniform;

[[group(1), binding(0)]]
var t_palette: texture_2d<f32>;
[[group(1), binding(1)]]
var s_palette: sampler;

[[stage(vertex)]]
fn vs_main(
	[[builtin(vertex_index)]] in_vertex_index: u32,
) -> VertexOutput {
	// this makes a square mesh.
	var out: VertexOutput;
	let x = f32(i32(!(in_vertex_index < 2u || in_vertex_index == 4u)) * 2 - 1);
	let y = f32(i32(!(in_vertex_index > 3u || in_vertex_index == 1u)) * 2 - 1);
	out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
	out.p = vec2<f32>(x, y);
	return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
	// assume Julia, z starts at frag position and c is set by data uniform
	var z: vec2<f32> = in.p * vec2<f32>(data.sclex, data.scley) + vec2<f32>(data.trslx, data.trsly);
	var c: vec2<f32> = vec2<f32>(data.cr, data.ci);
	// if mandelbrot, z starts at 0 + 0*i and c changes with frag position.
	if (data.mandel > 0u) {
		c = z;
		z = vec2<f32>(0.0, 0.0);
	}
	// iteration
	var n: u32 = 0u;
	loop {
		// if squared length exceed 2²
		if (dot(z, z) > 4.0 || n >= data.depth) { break; }
		// z² + c
		z = vec2<f32>(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y);
		// no ++ in wgsl
		n = n + 1u;
	}
	// basic coloring based on n
	let v = f32(n) / f32(data.depth);
	return textureSample(t_palette, s_palette, vec2<f32>(v, 0.5));
}
