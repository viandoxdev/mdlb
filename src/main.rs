use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder, Window},
};
use log::{
    debug,
    info
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DataUniform {
    cr: f32,
    ci: f32,
    // treat as boolean
    mandel: u32,
    depth: u32,
    sclex: f32,
    scley: f32,
    trslx: f32,
    trsly: f32
}

struct ColorPalette {
    gradient: Vec<wgpu::Color>,
}
impl ColorPalette {
    fn new(gradient: Vec<wgpu::Color>) -> Self {
        Self {
            gradient
        }
    }
    fn make_texture(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        let texture_size = wgpu::Extent3d {
            width: self.gradient.len() as u32,
            height: 1,
            depth_or_array_layers: 1
        };
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("palette_texture")
            }
        );
        let texture_rgba = self.gradient.clone()
            .iter().map(|v| vec![
                (v.r * 255.0) as u8,
                (v.g * 255.0) as u8,
                (v.b * 255.0) as u8,
                (v.a * 255.0) as u8])
            .flatten()
            .collect::<Vec<u8>>();
            
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            texture_rgba.as_slice(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * self.gradient.len() as u32),
                rows_per_image: std::num::NonZeroU32::new(1u32),
            },
            texture_size
        );

        texture
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    clear_color: wgpu::Color,
    data_uniform: DataUniform,
    data_buffer: wgpu::Buffer,
    data_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    scale: f32,
    mouse_down: bool,
    mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,
    ctrl_down: bool,
    fix_julia: bool,
    palette: ColorPalette,
    texture_bind_group: wgpu::BindGroup
}
impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // instace -> handle to gpu
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        #[cfg(not(target_arch = "wasm32"))]
        {
            instance.enumerate_adapters(wgpu::Backends::all()).for_each(|a| {
                let info = a.get_info();
                println!("Adapter {}:", info.name);
                println!("  backend: {:?}", info.backend);
                println!("  type: {:?}", info.device_type);
            });
        }
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false
            }
        ).await.expect("No suitable adapter (gpu) found, aborting.");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                label: None
            },
            None
        ).await.unwrap();
        info!("device OK");
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into())
        });
        info!("Shaders OK");

        let data_uniform = DataUniform {
            ci: 0.0,
            cr: 0.0,
            mandel: 1,
            depth: 100,
            trslx: 0.0,
            trsly: 0.0,
            sclex: 1.0,
            scley: 1.0
        };

        let data_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Data Buffer"),
                contents: bytemuck::cast_slice(&[data_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            }
        );

        let data_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                }
            ],
            label: Some("data_bind_group_layout")
        });

        let data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &data_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding()
                }
            ],
            label: Some("data_bind_group")
        });

        info!("data uniform and bind group OK");

        let palette = ColorPalette::new(
            // different palettes:
            // blue red
            //vec![
               //wgpu::Color {r: 0.000, g: 0.000, b: 0.000, a: 1.000},
               //wgpu::Color {r: 0.015, g: 0.082, b: 0.439, a: 1.000},
               //wgpu::Color {r: 0.070, g: 0.760, b: 0.913, a: 1.000},
               //wgpu::Color {r: 0.768, g: 0.443, b: 0.929, a: 1.000},
               //wgpu::Color {r: 0.972, g: 0.309, b: 0.349, a: 1.000},
               //wgpu::Color {r: 1.000, g: 1.000, b: 1.000, a: 1.000},
           //]
           // black and white
            vec![
               wgpu::Color {r: 0.000, g: 0.000, b: 0.000, a: 1.000},
               wgpu::Color {r: 1.000, g: 1.000, b: 1.000, a: 1.000},
           ]
           // // black white and black
            //vec![
               //wgpu::Color {r: 0.000, g: 0.000, b: 0.000, a: 1.000},
               //wgpu::Color {r: 1.000, g: 1.000, b: 1.000, a: 1.000},
               //wgpu::Color {r: 0.000, g: 0.000, b: 0.000, a: 1.000},
           //]
        );

        let texture = palette.make_texture(&device, &queue);

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float {filterable: true},
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true
                        },
                        count: None
                    }
                ],
                label: Some("texture_bind_group_layout")
            }
        );

        let texture_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view)
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler)
                    }
                ],
                label: Some("texture_bind_group")
            }
        );

        info!("palette texture and bind group OK");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &data_bind_group_layout,
                    &texture_bind_group_layout
                ],
                push_constant_ranges: &[]
            });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[]
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                clamp_depth: false,
                conservative: false
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            }
        });

        info!("render pipeline OK");
        info!("state initialization DONE");

        Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color: wgpu::Color{r: 0.0, g: 0.0, b: 0.0, a: 1.0},
            data_uniform,
            data_buffer,
            data_bind_group,
            render_pipeline,
            scale: 1.0,
            mouse_down: false,
            mouse_pos: None,
            ctrl_down: false,
            fix_julia: false,
            palette,
            texture_bind_group,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved {position, ..} => {
                match self.mouse_pos {
                    Some(pp) => {
                        let delta = winit::dpi::PhysicalPosition {x: position.x - pp.x, y: position.y - pp.y};
                        if self.mouse_down {
                            self.data_uniform.trslx -= delta.x as f32 / self.size.width as f32 * 2.0 * self.data_uniform.sclex;
                            self.data_uniform.trsly += delta.y as f32 / self.size.height as f32 * 2.0 * self.data_uniform.scley;
                        }
                        self.mouse_pos = Some(*position);
                    },
                    None => {
                        self.mouse_pos = Some(*position);
                    }
                }

                true
            },
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::F),
                    ..
                },
                ..
            } => {
                if self.data_uniform.mandel == 0 {
                    self.fix_julia = !self.fix_julia;
                }
                false
            },
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::M),
                    ..
                },
                ..
            } => {
                if self.data_uniform.mandel > 0 {
                    self.data_uniform.mandel = 0;
                } else {
                    self.data_uniform.mandel = 1;
                }
                false
            },
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::LControl),
                    ..
                },
                ..
            } => {
                self.ctrl_down = true;
                false
            },
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state: ElementState::Released,
                    virtual_keycode: Some(VirtualKeyCode::LControl),
                    ..
                },
                ..
            } => {
                self.ctrl_down = false;
                false
            },
            WindowEvent::MouseInput {state, button, ..} => {
                match button {
                    MouseButton::Left => {
                        self.mouse_down = *state == ElementState::Pressed;
                    },
                    _ => {}
                }
                false
            },
            WindowEvent::MouseWheel {delta, ..} => {
                let mut e = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        *y
                    },
                    MouseScrollDelta::PixelDelta(pos) => {
                        // magic number goes brbr
                        pos.y as f32 / 100.0
                    }
                };
                e *= 2.0;
                if self.ctrl_down {
                    if self.data_uniform.depth as i32 + e as i32 > 10 {
                        if e < 0.0 {
                            self.data_uniform.depth -= (-e) as u32;
                        } else {
                            self.data_uniform.depth += e as u32;
                        }
                        println!("depth changed: {}", self.data_uniform.depth);
                    }
                } else {
                    self.scale /= e.powf(e / e.abs()).abs();
                    // make sure scale > 0
                    self.scale = self.scale.max(f32::MIN_POSITIVE);
                }
                false
            }
            _ => false
        }
    }

    fn update(&mut self) {
        if self.size.width > self.size.height {
            self.data_uniform.scley = self.scale;
            self.data_uniform.sclex = self.size.width as f32 / self.size.height as f32 * self.scale;
        } else {
            self.data_uniform.sclex = self.scale;
            self.data_uniform.scley = self.size.height as f32 / self.size.width as f32 * self.scale;
        }
        if self.mouse_pos.is_some() && !self.fix_julia {
            self.data_uniform.cr = (self.mouse_pos.unwrap().x as f32 / self.size.width as f32 * 2.0 - 1.0) * self.data_uniform.sclex + self.data_uniform.trslx;
            self.data_uniform.ci = (self.mouse_pos.unwrap().y as f32 / self.size.height as f32 * 2.0 - 1.0) * self.data_uniform.scley - self.data_uniform.trsly;
        }
        self.queue.write_buffer(&self.data_buffer, 0, bytemuck::cast_slice(&[self.data_uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        info!(" --- RENDER BEGIN --- ");

        let output = self.surface.get_current_texture().expect("could not get render texture");
        info!("render texture OK");
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        info!("render texture view OK");

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        info!("encoder OK");

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    }
                }],
                depth_stencil_attachment: None
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.data_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);

            render_pass.draw(0..6, 0..1);

            info!("render pass OK");
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        info!("present OK");
        Ok(())
    }
}

async fn run(window: Window, event_loop: EventLoop<()>) {
    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::RedrawRequested(_) => {
            state.update();
            match state.render() {
                Ok(_) => {},
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        },
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => if !state.input(event) {
            match event {
                WindowEvent::CloseRequested 
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                },
                WindowEvent::ScaleFactorChanged { new_inner_size, ..} => {
                    state.resize(**new_inner_size);
                },
                _ => {}
            }
        }
        _ => {}
        
    });
}

fn main() {
    info!("hello there");
    //env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut state: State; 
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(window, event_loop));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Error).expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(window, event_loop));
    }

}
