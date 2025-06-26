#![feature(array_try_from_fn)]

use std::{array, error};
use std::fmt::{self, Debug};
use std::io::{self, Read, Write};
use std::string;

#[derive(Debug)]
pub struct XnbData {
    pub target_platform: TargetPlatform,
    pub format_version: u8,
    pub flag_bits: u8,
    pub shared_resources: Vec<SharedResource>,
}

impl TryFrom<&mut dyn Read> for XnbData {
    type Error = XnbError;
    fn try_from(data: &mut dyn Read) -> Result<Self, Self::Error> {
        let mut format_identifier = [0u8; 3];
        data.read_exact(&mut format_identifier).map_err(XnbError::Io)?;
        if !matches!(format_identifier, [b'X', b'N', b'B']) {
            return Err(XnbError::InvalidFileFormat(format_identifier));
        }

        let target_platform: TargetPlatform = read_u8(data)?.try_into()?;
        if target_platform != TargetPlatform::Windows {
            return Err(XnbError::UnimplementedPlatform(target_platform));
        }

        let format_version = read_u8(data)?;
        if format_version != 5 {
            return Err(XnbError::UnimplementedVersion(format_version));
        }

        let flag_bits = read_u8(data)?;
        let is_compressed = flag_bits & 0x80 == 1;
        let _is_hi_def_profile = flag_bits & 0x01 == 1;

        // TODO: XMemCompress
        if is_compressed {
            return Err(XnbError::UnimplementedCompression);
        }

        let _compressed_file_size = read_u32(data)?;

        // TODO: XMemCompress
        if is_compressed {
            let _decompressed_data_size = read_u32(data)?;
            unreachable!();
        }

        let mut type_reader_count = read_seven_bit_encoded_int(data)?;
        if type_reader_count == 0 {
            return Err(XnbError::NoTypeReaders);
        }

        let mut readers = Vec::with_capacity(type_reader_count as usize);
        while type_reader_count != 0 {
            let type_reader_name = read_string(data)?;
            let reader_version_number = read_i32(data)?;

            readers.push((type_reader_name, reader_version_number));
            type_reader_count -= 1;
        }
        println!("Type readers: {readers:?}");

        let mut shared_resource_count = read_seven_bit_encoded_int(data)?;
        shared_resource_count += 1;

        let mut shared_resources = Vec::with_capacity(shared_resource_count as usize);
        for _ in 0..shared_resource_count {
            let shared_resource = read_shared_resource(&readers, data)?;
            shared_resources.push(shared_resource);
        }

        Ok(Self {
            target_platform,
            format_version,
            flag_bits,
            shared_resources,
        })
    }
}

#[derive(Debug)]
pub struct Matrix{ 
    pub m11: f32, pub m12: f32, pub m13: f32, pub m14: f32,
    pub m21: f32, pub m22: f32, pub m23: f32, pub m24: f32,
    pub m31: f32, pub m32: f32, pub m33: f32, pub m34: f32,
    pub m41: f32, pub m42: f32, pub m43: f32, pub m44: f32,
}

#[derive(Debug)]
pub struct Vector2{ pub x: f32, pub y: f32 }

#[derive(Debug)]
pub struct Vector3{ pub x: f32, pub y: f32, pub z: f32 }

#[derive(Debug)]
pub struct Vector4{ pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

#[derive(Debug)]
pub struct Quaternion{ pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

#[derive(Debug)]
pub struct Decimal(pub u32, pub u32, pub u32, pub u32);

#[derive(Debug)]
pub struct Color{ pub r: u8, pub g: u8, pub b: u8, pub a: u8 }

#[derive(Debug)]
pub struct Plane{ pub v: Vector3, pub d: f32 }

#[derive(Debug)]
pub struct Rectangle{ pub pt: Vector2, pub w: i32, pub h: i32 }

#[derive(Debug)]
pub struct BoundingBox{ pub min: Vector3, pub max: Vector3 }

#[derive(Debug)]
pub struct BoundingSphere{ pub center: Vector3, pub r: f32 }

#[derive(Debug)]
pub struct Ray{ pub pt: Vector3, pub dir: Vector3 }

#[derive(Debug)]
pub struct Curve{ pub pre: CurveLoop, pub post: CurveLoop, pub keys: Vec<Key>, }

#[derive(Debug)]
pub enum CurveLoop {
    Constant,
    Cycle,
    CycleOffset,
    Oscillate,
    Linear,
}

impl CurveLoop {
    fn new(n: i32) -> Option<Self> {
        match n {
            0 => Some(Self::Constant),
            1 => Some(Self::Cycle),
            2 => Some(Self::CycleOffset),
            3 => Some(Self::Oscillate),
            4 => Some(Self::Linear),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum SurfaceFormat {
    Color,
    Bgr565,
    Bgra5551,
    Bgra4444,
    Dxt1,
    Dxt3,
    Dxt5,
    NormalizedByte2,
    NormalizedByte4,
    Rgba1010102,
    Rg32,
    Rgba64,
    Alpha8,
    Single,
    Vector2,
    Vector4,
    HalfSingle,
    HalfVector2,
    HalfVector4,
    HdrBlendable,
}

impl SurfaceFormat {
    fn new(n: i32) -> Option<Self> {
        match n {
            0 => Some(Self::Color),
            1 => Some(Self::Bgr565),
            2 => Some(Self::Bgra5551),
            3 => Some(Self::Bgra4444),
            4 => Some(Self::Dxt1),
            5 => Some(Self::Dxt3),
            6 => Some(Self::Dxt5),
            7 => Some(Self::NormalizedByte2),
            8 => Some(Self::NormalizedByte4),
            9 => Some(Self::Rgba1010102),
           10 => Some(Self::Rg32),
           11 => Some(Self::Rgba64),
           12 => Some(Self::Alpha8),
           13 => Some(Self::Single),
           14 => Some(Self::Vector2),
           15 => Some(Self::Vector4),
           16 => Some(Self::HalfSingle),
           17 => Some(Self::HalfVector2),
           18 => Some(Self::HalfVector4),
           19 => Some(Self::HdrBlendable),
            _ => None,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Mip {
    data: Vec<u8>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Texture2d {
    surface_format: SurfaceFormat,
    width: u32,
    height: u32,
    mips: Vec<Mip>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Texture3d {
    surface_format: SurfaceFormat,
    width: u32,
    height: u32,
    depth: u32,
    mips: Vec<Mip>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct TextureCube {
    surface_format: SurfaceFormat,
    size: u32,
    faces: [Vec<Mip>; 6],
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct IndexBuffer {
    is_16_bit: bool,
    data: Vec<u8>,
}

#[derive(Debug)]
pub enum ElemFormat {
    Single,
    Vector2,
    Vector3,
    Vector4,
    Color,
    Byte4,
    Short2,
    Short4,
    NormalizedShort2,
    NormalizedShort4,
    HalfVector2,
    HalfVector4,
}
impl ElemFormat {
    fn new(n: i32) -> Option<Self> {
        match n {
            0 => Some(Self::Single),
            1 => Some(Self::Vector2),
            2 => Some(Self::Vector3),
            3 => Some(Self::Vector4),
            4 => Some(Self::Color),
            5 => Some(Self::Byte4),
            6 => Some(Self::Short2),
            7 => Some(Self::Short4),
            8 => Some(Self::NormalizedShort2),
            9 => Some(Self::NormalizedShort4),
           10 => Some(Self::HalfVector2),
           11 => Some(Self::HalfVector4),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum ElemUsage {
    Position,
    Color,
    TextureCoordinate,
    Normal,
    Binormal,
    Tangent,
    BlendIndices,
    BlendWeight,
    Depth,
    Fog,
    PointSize,
    Sample,
    TesselateFactor,
}
impl ElemUsage {
    fn new(n: i32) -> Option<Self> {
        match n {
            0 => Some(Self::Position),
            1 => Some(Self::Color),
            2 => Some(Self::TextureCoordinate),
            3 => Some(Self::Normal),
            4 => Some(Self::Binormal),
            5 => Some(Self::Tangent),
            6 => Some(Self::BlendIndices),
            7 => Some(Self::BlendWeight),
            8 => Some(Self::Depth),
            9 => Some(Self::Fog),
           10 => Some(Self::PointSize),
           11 => Some(Self::Sample),
           12 => Some(Self::TesselateFactor),
            _ => None,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Elem {
    offset: u32,
    format: ElemFormat,
    usage: ElemUsage,
    usage_index: u32,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct VertexDecl {
    stride: u32,
    elems: Vec<Elem>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct VertexBuffer {
    decl: VertexDecl,
    vertex_count: u32,
    vertex_data: Vec<u8>,
}

#[derive(Debug)]
pub enum SharedResource {
    Null,
    Byte(u8),
    SByte(i8),
    Int16(i16),
    UInt16(u16),
    Int32(i32),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
    Single(f32),
    Double(f64),
    Boolean(bool),
    Char(char),
    String(String),
    //Enum(u32),
    //Array(Vec<Resource>),
    //List(Vec<Resource>),
    //Dictionary(BTreeMap<Resource, Resource>),
    TimeSpan { tick_count: i64 },
    DateTime { packed_value: u64 },
    Decimal(Decimal),
    ExternalReference(String),

    Vector2(Vector2),
    Vector3(Vector3),
    Vector4(Vector4),
    Matrix(Matrix),
    Quaternion(Quaternion),
    Color(Color),
    Plane(Plane),
    Point(Vector2),
    Rectangle(Rectangle),
    BoundingBox(BoundingBox),
    BoundingSphere(BoundingSphere),
    BoundingFrustum(Matrix),
    Ray(Ray),
    Curve(Curve),

    Texture2d(Texture2d),
    Texture3d(Texture3d),
    TextureCube(TextureCube),
    IndexBuffer(IndexBuffer),
    VertexBuffer(VertexBuffer),
    VertexDecl(VertexDecl),

    // TODO: below
    //Effect(Effect),
    //Material(Material),
    //BasicEffect(BasicEffect),
    //AlphaTestEffect(AlphaTestEffect),

    SoundEffect(SoundEffect),
    Unsupported(String),
}

#[derive(Debug)]
pub enum Continuity {
    Smooth,
    Step,
}

impl Continuity {
    fn new(n: i32) -> Option<Self> {
        match n {
            0 => Some(Self::Smooth),
            1 => Some(Self::Step),
            _ => None,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Key {
    position: f32,
    value: f32,
    tan_in: f32,
    tan_out: f32,
    continuity: Continuity,
}

fn read_shared_resource(
    type_readers: &[(String, i32)],
    r: &mut dyn Read
) -> Result<SharedResource, XnbError> {
    let reader_id = read_seven_bit_encoded_int(r)?;
    if reader_id == 0 {
        return Ok(SharedResource::Null);
    }
    let reader_id = reader_id - 1;
    
    match type_readers
            .get(reader_id as usize)
            .ok_or(XnbError::InvalidTypeReaderId(reader_id))?.0.as_str() {
        "Microsoft.Xna.Framework.Content.ByteReader"
            => Ok(SharedResource::Byte(read_u8(r)?)),
        "Microsoft.Xna.Framework.Content.SByteReader"
            => Ok(SharedResource::SByte(read_i8(r)?)),
        "Microsoft.Xna.Framework.Content.Int16Reader"
            => Ok(SharedResource::Int16(read_i16(r)?)),
        "Microsoft.Xna.Framework.Content.UInt16Reader"
            => Ok(SharedResource::UInt16(read_u16(r)?)),
        "Microsoft.Xna.Framework.Content.Int32Reader"
            => Ok(SharedResource::Int32(read_i32(r)?)),
        "Microsoft.Xna.Framework.Content.UInt32Reader"
            => Ok(SharedResource::UInt32(read_u32(r)?)),
        "Microsoft.Xna.Framework.Content.Int64Reader"
            => Ok(SharedResource::Int64(read_i64(r)?)),
        "Microsoft.Xna.Framework.Content.UInt64Reader"
            => Ok(SharedResource::UInt64(read_u64(r)?)),
        "Microsoft.Xna.Framework.Content.SingleReader"
            => Ok(SharedResource::Single(read_f32(r)?)),
        "Microsoft.Xna.Framework.Content.DoubleReader"
            => Ok(SharedResource::Double(read_f64(r)?)),
        "Microsoft.Xna.Framework.Content.BooleanReader"
            => Ok(SharedResource::Boolean(read_bool(r)?)),
        "Microsoft.Xna.Framework.Content.CharReader"
            => Ok(SharedResource::Char(read_char(r)?)),
        "Microsoft.Xna.Framework.Content.StringReader"
            => Ok(SharedResource::String(read_string(r)?)),
        "Microsoft.Xna.Framework.Content.TimeSpanReader"
            => Ok(SharedResource::TimeSpan { tick_count: read_i64(r)? }),
        "Microsoft.Xna.Framework.Content.DateTimeReader"
            => Ok(SharedResource::DateTime { packed_value: read_u64(r)? }),
        "Microsoft.Xna.Framework.Content.ExternalReferenceReader" 
            => Ok(SharedResource::ExternalReference(read_string(r)?)),
        "Microsoft.Xna.Framework.Content.DecimalReader" 
            => Ok(SharedResource::Decimal(read_decimal(r)?)),

        "Microsoft.Xna.Framework.Content.Vector2Reader" 
            => Ok(SharedResource::Vector2(read_vector2(r)?)),
        "Microsoft.Xna.Framework.Content.Vector3Reader" 
            => Ok(SharedResource::Vector3(read_vector3(r)?)),
        "Microsoft.Xna.Framework.Content.Vector4Reader" 
            => Ok(SharedResource::Vector4(read_vector4(r)?)),
        "Microsoft.Xna.Framework.Content.MatrixReader" 
            => Ok(SharedResource::Matrix(read_matrix(r)?)),
        "Microsoft.Xna.Framework.Content.QuaternionReader" 
            => Ok(SharedResource::Quaternion(read_quaternion(r)?)),
        "Microsoft.Xna.Framework.Content.ColorReader" 
            => Ok(SharedResource::Color(read_color(r)?)),
        "Microsoft.Xna.Framework.Content.PlaneReader" 
            => Ok(SharedResource::Plane(read_plane(r)?)),
        "Microsoft.Xna.Framework.Content.PointReader" 
            => Ok(SharedResource::Point(read_vector2(r)?)),
        "Microsoft.Xna.Framework.Content.RectangleReader" 
            => Ok(SharedResource::Rectangle(read_rectangle(r)?)),
        "Microsoft.Xna.Framework.Content.BoundingBoxReader" 
            => Ok(SharedResource::BoundingBox(read_bounding_box(r)?)),
        "Microsoft.Xna.Framework.Content.BoundingSphereReader" 
            => Ok(SharedResource::BoundingSphere(read_bounding_sphere(r)?)),
        "Microsoft.Xna.Framework.Content.BoundingFrustumReader" 
            => Ok(SharedResource::BoundingFrustum(read_bounding_frustum(r)?)),
        "Microsoft.Xna.Framework.Content.RayReader" 
            => Ok(SharedResource::Ray(read_ray(r)?)),
        "Microsoft.Xna.Framework.Content.CurveReader" 
            => Ok(SharedResource::Curve(read_curve(r)?)),

        "Microsoft.Xna.Framework.Content.Texture2DReader" 
            => Ok(SharedResource::Texture2d(read_texure2d(r)?)),
        "Microsoft.Xna.Framework.Content.Texture3DReader" 
            => Ok(SharedResource::Texture3d(read_texure3d(r)?)),
        "Microsoft.Xna.Framework.Content.TextureCubeReader" 
            => Ok(SharedResource::TextureCube(read_texure_cube(r)?)),
        "Microsoft.Xna.Framework.Content.IndexBufferReader" 
            => Ok(SharedResource::IndexBuffer(read_index_buffer(r)?)),
        "Microsoft.Xna.Framework.Content.VertexBufferReader" 
            => Ok(SharedResource::VertexBuffer(read_vertex_buffer(r)?)),
        "Microsoft.Xna.Framework.Content.VertexDeclReader" 
            => Ok(SharedResource::VertexDecl(read_vertex_decl(r)?)),

        // TODO:
        // effects

        "Microsoft.Xna.Framework.Content.SoundEffectReader" => read_sound_effect(r),

        name => {
            Ok(SharedResource::Unsupported(name.to_string()))
        },
    }
}

fn read_vector2(r: &mut dyn Read) -> Result<Vector2, XnbError> {
    Ok(Vector2 { x: read_f32(r)?, y: read_f32(r)? })
}

fn read_vector3(r: &mut dyn Read) -> Result<Vector3, XnbError> {
    Ok(Vector3 { x: read_f32(r)?, y: read_f32(r)?, z: read_f32(r)? })
}

fn read_vector4(r: &mut dyn Read) -> Result<Vector4, XnbError> {
    Ok(Vector4 { x: read_f32(r)?, y: read_f32(r)?, z: read_f32(r)?, w: read_f32(r)? })
}

fn read_matrix(r: &mut dyn Read) -> Result<Matrix, XnbError> {
    Ok(Matrix {
        m11: read_f32(r)?, m12: read_f32(r)?, m13: read_f32(r)?, m14: read_f32(r)?,
        m21: read_f32(r)?, m22: read_f32(r)?, m23: read_f32(r)?, m24: read_f32(r)?,
        m31: read_f32(r)?, m32: read_f32(r)?, m33: read_f32(r)?, m34: read_f32(r)?,
        m41: read_f32(r)?, m42: read_f32(r)?, m43: read_f32(r)?, m44: read_f32(r)?,
    })
}

fn read_quaternion(r: &mut dyn Read) -> Result<Quaternion, XnbError> {
    Ok(Quaternion {
        x: read_f32(r)?, y: read_f32(r)?, z: read_f32(r)?, w: read_f32(r)?,
    })
}

fn read_color(r: &mut dyn Read) -> Result<Color, XnbError> {
    Ok(Color {
        r: read_u8(r)?,
        g: read_u8(r)?,
        b: read_u8(r)?,
        a: read_u8(r)?,
    })
}

fn read_decimal(r: &mut dyn Read) -> Result<Decimal, XnbError> {
    Ok(Decimal(
        read_u32(r)?,
        read_u32(r)?,
        read_u32(r)?,
        read_u32(r)?,
    ))
}

fn read_plane(r: &mut dyn Read) -> Result<Plane, XnbError> {
    Ok(Plane {
        v: read_vector3(r)?,
        d: read_f32(r)?,
    })
}

fn read_rectangle(r: &mut dyn Read) -> Result<Rectangle, XnbError> {
    Ok(Rectangle {
        pt: read_vector2(r)?,
        w: read_i32(r)?,
        h: read_i32(r)?
    })
}

fn read_bounding_box(r: &mut dyn Read) -> Result<BoundingBox, XnbError> {
    Ok(BoundingBox {
        min: read_vector3(r)?,
        max: read_vector3(r)?,
    })
}

fn read_bounding_sphere(r: &mut dyn Read) -> Result<BoundingSphere, XnbError> {
    Ok(BoundingSphere {
        center: read_vector3(r)?,
        r: read_f32(r)?,
    })
}

fn read_bounding_frustum(r: &mut dyn Read) -> Result<Matrix, XnbError> {
    Ok(read_matrix(r)?)
}

fn read_ray(r: &mut dyn Read) -> Result<Ray, XnbError> {
    Ok(Ray {
        pt: read_vector3(r)?,
        dir: read_vector3(r)?,
    })
}

fn read_key(r: &mut dyn Read) -> Result<Key, XnbError> {
    let position = read_f32(r)?;
    let value = read_f32(r)?;
    let tan_in = read_f32(r)?;
    let tan_out = read_f32(r)?;
    let continuity = read_i32(r)?;
    let continuity = Continuity::new(continuity)
        .ok_or(XnbError::InvalidContinuityValue(continuity))?;
    Ok(Key {
        position,
        value,
        tan_in,
        tan_out,
        continuity,
    })
}

fn read_curve(r: &mut dyn Read) -> Result<Curve, XnbError> {
    let pre_loop = read_i32(r)?;
    let pre_loop = CurveLoop::new(pre_loop).ok_or(XnbError::InvalidCurveLoopValue(pre_loop))?;
    let post_loop = read_i32(r)?;
    let post_loop = CurveLoop::new(post_loop).ok_or(XnbError::InvalidCurveLoopValue(post_loop))?;
    let key_count = read_u32(r)?;

    let mut keys = Vec::with_capacity(key_count as usize);
    for _ in 0..key_count {
        let key = read_key(r)?;
        keys.push(key);
    }

    Ok(Curve {
        pre: pre_loop,
        post: post_loop,
        keys 
    })
}

fn read_mip(r: &mut dyn Read) -> Result<Mip , XnbError> {
    let data_sz = read_u32(r)?;
    let mut data = vec![0u8; data_sz as usize];
    r.read_exact(&mut data).map_err(XnbError::Io)?;
    Ok(Mip { data })
}

fn read_texure2d(r: &mut dyn Read) -> Result<Texture2d, XnbError> {
    let surface_format = read_i32(r)?;
    let surface_format = SurfaceFormat::new(surface_format).ok_or(XnbError::InvalidSurfaceFormat(surface_format))?;
    let width = read_u32(r)?;
    let height = read_u32(r)?;

    let mip_count = read_u32(r)?;
    let mut mips = Vec::with_capacity(mip_count as usize);
    for _ in 0..mip_count {
        let mip = read_mip(r)?;
        mips.push(mip);
    }

    Ok(Texture2d { surface_format, width, height, mips })
}

fn read_texure3d(r: &mut dyn Read) -> Result<Texture3d, XnbError> {
    let surface_format = read_i32(r)?;
    let surface_format = SurfaceFormat::new(surface_format).ok_or(XnbError::InvalidSurfaceFormat(surface_format))?;
    let width = read_u32(r)?;
    let height = read_u32(r)?;
    let depth = read_u32(r)?;

    let mip_count = read_u32(r)?;
    let mut mips = Vec::with_capacity(mip_count as usize);
    for _ in 0..mip_count {
        let mip = read_mip(r)?;
        mips.push(mip);
    }

    Ok(Texture3d { surface_format, width, height, depth, mips })
}

fn read_texure_cube(r: &mut dyn Read) -> Result<TextureCube, XnbError> {
    let surface_format = read_i32(r)?;
    let surface_format = SurfaceFormat::new(surface_format).ok_or(XnbError::InvalidSurfaceFormat(surface_format))?;
    let size = read_u32(r)?;

    let mip_count = read_u32(r)?;

    let faces = array::try_from_fn(|_| {
        let mut mips = Vec::with_capacity(mip_count as usize);
        for _ in 0..mip_count {
            let mip = read_mip(r)?;
            mips.push(mip);
        }
        Ok(mips)
    })?;

    Ok(TextureCube {
        surface_format,
        size,
        faces,
    })
}

fn read_index_buffer(r: &mut dyn Read) -> Result<IndexBuffer, XnbError> {
    let is_16_bit = read_bool(r)?;
    let data_sz = read_u32(r)?;
    let mut data = vec![0; data_sz as usize];
    r.read_exact(&mut data).map_err(XnbError::Io)?;
    Ok(IndexBuffer { is_16_bit, data })
}

fn read_elem(r: &mut dyn Read) -> Result<Elem, XnbError> {
    let offset = read_u32(r)?;
    let format = read_i32(r)?;
    let format = ElemFormat::new(format).ok_or(XnbError::InvalidElementFormat(format))?;
    let usage = read_i32(r)?;
    let usage = ElemUsage::new(usage).ok_or(XnbError::InvalidElementUsage(usage))?;
    let usage_index = read_u32(r)?;
    Ok(Elem { offset, format, usage, usage_index })
}

fn read_vertex_decl(r: &mut dyn Read) -> Result<VertexDecl, XnbError> {
    let stride = read_u32(r)?;
    let elem_count = read_u32(r)?;
    let mut elems = Vec::with_capacity(elem_count as usize);
    for _ in 0..elem_count {
        let elem = read_elem(r)?;
        elems.push(elem);
    }
    Ok(VertexDecl { stride, elems })
}

fn read_vertex_buffer(r: &mut dyn Read) -> Result<VertexBuffer, XnbError> {
    let decl = read_vertex_decl(r)?;
    let vertex_count = read_u32(r)?;
    let mut vertex_data = vec![0; (vertex_count * decl.stride) as usize];
    r.read_exact(&mut vertex_data).map_err(XnbError::Io)?;
    Ok(VertexBuffer { decl, vertex_count, vertex_data })
}

fn read_sound_effect(r: &mut dyn Read) -> Result<SharedResource, XnbError> {
    let format_size = read_u32(r)?;
    if format_size != 18 {
        return Err(XnbError::InvalidFormatSize(format_size));
    }
    let waveformatex = read_wave_format_ex(r)?;

    let data_size = read_u32(r)?;
    if data_size == 0 {
        return Err(XnbError::InvalidDataSize(data_size));
    }
    let mut waveformdata = vec![0u8; data_size as usize];
    read_buf(r, &mut waveformdata)?;

    let loop_start = read_i32(r)?;
    let loop_len = read_i32(r)?;
    let duration_ms = read_i32(r)?;

    Ok(SharedResource::SoundEffect(SoundEffect {
        waveformatex,
        waveformdata,
        loop_start,
        loop_len,
        duration_ms,
    }))
}

pub struct SoundEffect {
    waveformatex: WaveFormatEx,
    waveformdata: Vec<u8>,
    loop_start: i32,
    loop_len: i32,
    duration_ms: i32,
}

impl SoundEffect {
    pub fn file_data(&self) -> Result<Vec<u8>, XnbError> {
        let mut data: Vec<u8> = Vec::with_capacity(42 + self.waveformdata.len());
        data.write("RIFF".as_bytes()).map_err(XnbError::Io)?;
        data.write(&(self.waveformdata.len() as u32).to_le_bytes()).map_err(XnbError::Io)?;
        data.write(b"WAVE").map_err(XnbError::Io)?;
        data.write(b"fmt ").map_err(XnbError::Io)?;
        data.write(&16u32.to_le_bytes()).map_err(XnbError::Io)?;
        data.write(&self.waveformatex.w_format_tag.to_le_bytes()).map_err(XnbError::Io)?;
        data.write(&self.waveformatex.n_channels.to_le_bytes()).map_err(XnbError::Io)?;
        data.write(&self.waveformatex.n_samples_per_sec.to_le_bytes()).map_err(XnbError::Io)?;
        data.write(&self.waveformatex.n_avg_bytes_per_sec.to_le_bytes()).map_err(XnbError::Io)?;
        data.write(&self.waveformatex.n_block_align.to_le_bytes()).map_err(XnbError::Io)?;
        data.write(&self.waveformatex.w_bits_per_sample.to_le_bytes()).map_err(XnbError::Io)?;
        data.write(b"data").map_err(XnbError::Io)?;
        data.write(&(self.waveformdata.len() as u32).to_le_bytes()).map_err(XnbError::Io)?;
        data.write(&self.waveformdata).map_err(XnbError::Io)?;
        Ok(data)
    }
}

impl Debug for SoundEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SoundEffect {{\r\n\t{:?},\r\n\twaveformdata_size: {},\r\n\tloop_start: {},\r\n\tloop_len: {},\r\n\tduration_ms: {}\r\n}}",
            self.waveformatex,
            self.waveformdata.len(),
            self.loop_start,
            self.loop_len,
            self.duration_ms,
        )
    }
}

// https://learn.microsoft.com/en-us/windows/win32/api/mmeapi/ns-mmeapi-waveformatex
#[allow(dead_code)]
#[derive(Debug)]
struct WaveFormatEx {
    w_format_tag: u16,
    n_channels: u16,
    n_samples_per_sec: u32,
    n_avg_bytes_per_sec: u32,
    n_block_align: u16,
    w_bits_per_sample: u16,
    cb_size: u16,
}

fn read_wave_format_ex(r: &mut dyn Read) -> Result<WaveFormatEx, XnbError> {
    let w_format_tag = read_u16(r)?;
    if w_format_tag != 1 {
        return Err(XnbError::UnimplementedWavCodec(w_format_tag));
    }
    Ok(WaveFormatEx {
        w_format_tag,
        n_channels: read_u16(r)?,
        n_samples_per_sec: read_u32(r)?,
        n_avg_bytes_per_sec: read_u32(r)?,
        n_block_align: read_u16(r)?,
        w_bits_per_sample: read_u16(r)?,
        cb_size: read_u16(r)?,
    })
}

fn read_buf(r: &mut dyn Read, buf: &mut [u8]) -> Result<(), XnbError> {
    r.read_exact(buf).map_err(XnbError::Io)
}

fn read_until_len_or_null(r: &mut dyn Read, len: usize) -> Result<Vec<u8>, XnbError> {
    let mut cursor = 0usize;
    let mut value = [0u8; 1];
    let mut data = vec![];
    loop {
        if cursor == len {
            break;
        }
        r.read_exact(&mut value).map_err(XnbError::Io)?;
        if value[0] == 0 {
            break;
        }
        cursor += 1;
        data.push(value[0]);
    }
    Ok(data)
}

fn read_bool(r: &mut dyn Read) -> Result<bool, XnbError> {
    Ok(read_u8(r)? == 1)
}

fn read_u8(r: &mut dyn Read) -> Result<u8, XnbError> {
    let mut num_data = [0u8; 1];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(u8::from_le_bytes(num_data))
}

fn read_i8(r: &mut dyn Read) -> Result<i8, XnbError> {
    let mut num_data = [0u8; 1];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(i8::from_le_bytes(num_data))
}

fn read_u16(r: &mut dyn Read) -> Result<u16, XnbError> {
    let mut num_data = [0u8; 2];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(u16::from_le_bytes(num_data))
}

fn read_i16(r: &mut dyn Read) -> Result<i16, XnbError> {
    let mut num_data = [0u8; 2];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(i16::from_le_bytes(num_data))
}

fn read_u32(r: &mut dyn Read) -> Result<u32, XnbError> {
    let mut num_data = [0u8; 4];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(u32::from_le_bytes(num_data))
}

fn read_i32(r: &mut dyn Read) -> Result<i32, XnbError> {
    let mut num_data = [0u8; 4];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(i32::from_le_bytes(num_data))
}

fn read_u64(r: &mut dyn Read) -> Result<u64, XnbError> {
    let mut num_data = [0u8; 8];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(u64::from_le_bytes(num_data))
}

fn read_i64(r: &mut dyn Read) -> Result<i64, XnbError> {
    let mut num_data = [0u8; 8];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(i64::from_le_bytes(num_data))
}

fn read_f32(r: &mut dyn Read) -> Result<f32, XnbError> {
    let mut num_data = [0u8; 4];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(f32::from_le_bytes(num_data))
}

fn read_f64(r: &mut dyn Read) -> Result<f64, XnbError> {
    let mut num_data = [0u8; 8];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(f64::from_le_bytes(num_data))
}

fn read_char(r: &mut dyn Read) -> Result<char, XnbError> {
    let num = read_u32(r)?;
    Ok(char::from_u32(num).ok_or(XnbError::InvalidChar(num))?)
}

fn read_string(r: &mut dyn Read) -> Result<String, XnbError> {
    let string_len = read_seven_bit_encoded_int(r)?;
    let string_data = read_until_len_or_null(r, string_len as usize)?;
    String::from_utf8(string_data).map_err(XnbError::Utf8)
}

// Rewrite of: https://github.com/dotnet/runtime/blob/5535e31a712343a63f5d7d796cd874e563e5ac14/src/libraries/System.Private.CoreLib/src/System/IO/BinaryReader.cs#L535C9-L577C10
fn read_seven_bit_encoded_int(r: &mut dyn Read) -> Result<i32, XnbError> {
    let mut result = 0u32;
    let mut value = [0u8; 1];
    let max_bytes_without_overflow = 4;

    let mut shift = 0u32;
    loop {
        r.read_exact(&mut value).map_err(XnbError::Io)?;
        result |= ((value[0] & 0x7fu8) as u32).wrapping_shl(shift);
        if value[0] <= 0x7fu8 {
            return Ok(result as i32);
        }
        if shift < max_bytes_without_overflow * 7 {
            break;
        }
        shift += 7;
    }

    r.read_exact(&mut value).map_err(XnbError::Io)?;
    if value[0] > 0b1111u8 {
        return Err(XnbError::Invalid7BitInt(value[0]));
    }
    result |= (value[0] as u32).wrapping_shl(max_bytes_without_overflow * 7);

    Ok(result as i32)
}

#[repr(u8)]
#[derive(Debug, PartialEq)]
pub enum TargetPlatform {
    Windows = b'w',
    WindowsPhone = b'm',
    Xbox = b'x',
}

impl TryFrom<u8> for TargetPlatform {
    type Error = XnbError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'w' => Ok(TargetPlatform::Windows),
            b'm' => Ok(TargetPlatform::WindowsPhone),
            b'x' => Ok(TargetPlatform::Xbox),
            _ => Err(XnbError::InvalidPlatform(value))
        }
    }
}

#[derive(Debug)]
pub enum XnbError {
    Io(io::Error),
    Utf8(string::FromUtf8Error),
    InvalidChar(u32),
    Invalid7BitInt(u8),
    InvalidFileFormat([u8; 3]),
    InvalidPlatform(u8),
    InvalidTypeReaderId(i32),
    InvalidFormatSize(u32),
    InvalidDataSize(u32),
    UnimplementedVersion(u8),
    UnimplementedPlatform(TargetPlatform),
    UnimplementedCompression,
    NoTypeReaders,
    NoSharedResources,
    UnimplementedWavCodec(u16),
    InvalidContinuityValue(i32),
    InvalidCurveLoopValue(i32),
    InvalidSurfaceFormat(i32),
    InvalidElementFormat(i32),
    InvalidElementUsage(i32),
}

impl fmt::Display for XnbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for XnbError { }

