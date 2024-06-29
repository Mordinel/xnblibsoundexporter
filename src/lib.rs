#![allow(dead_code)]
use std::{error, fmt::{self, Debug}, fs::File, io::{self, BufWriter, Read, Write}, string};

#[derive(Debug)]
pub struct XnbData {
    target_platform: TargetPlatform,
    format_version: u8,
    flag_bits: u8,
    shared_resources: Vec<SharedResource>,
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
        //if flag_bits != 0 {
        //    return Err(XnbError::UnimplementedProfile(flag_bits));
        //}

        let _compressed_file_size = read_u32(data)?;

        let mut type_reader_count = read_seven_bit_encoded_int(data)?;
        if type_reader_count == 0 {
            return Err(XnbError::NoTypeReaders);
        }

        let mut readers = Vec::with_capacity(type_reader_count as usize);
        while type_reader_count != 0 {
            let type_reader_name = read_dotnet_string(data)?;
            let reader_version_number = read_i32(data)?;

            readers.push((type_reader_name, reader_version_number));
            type_reader_count -= 1;
        }
        println!("Type readers: {readers:?}");

        let mut shared_resource_count = read_seven_bit_encoded_int(data)?;
        //if shared_resource_count == 0 {
        //    return Err(XnbError::NoSharedResources);
        //}
        shared_resource_count += 1;

        let mut shared_resources = Vec::with_capacity(shared_resource_count as usize);
        while shared_resource_count > 0 {
            let shared_resource = read_shared_resource(&readers, data)?;
            'write_file: {
                if let SharedResource::SoundEffect(ref sfx) = shared_resource {
                    let file_name = format!("resource_{shared_resource_count}.wav");
                    let file = File::create_new(&file_name);
                    if let Err(e) = file {
                        eprintln!("ERROR: {e}");
                        break 'write_file;
                    }
                    let file = file.unwrap();
                    let mut writer = BufWriter::new(file);
                    println!("Writing sound effect to file `{file_name}`");
                    write_wav(&mut writer, sfx);
                }
            }
            shared_resources.push(shared_resource);
            shared_resource_count -= 1;
        }

        Ok(Self {
            target_platform,
            format_version,
            flag_bits,
            shared_resources,
        })
    }
}

fn write_wav(w: &mut dyn Write, sfx: &SoundEffect) {
    let _ = w.write_all("RIFF".as_bytes());
    let _ = w.write_all(&(sfx.waveformdata.len() as u32).to_le_bytes());
    let _ = w.write_all("WAVE".as_bytes());
    let _ = w.write_all("fmt ".as_bytes());
    let _ = w.write_all(&16u32.to_le_bytes());
    let _ = w.write_all(&sfx.waveformatex.w_format_tag.to_le_bytes());
    let _ = w.write_all(&sfx.waveformatex.n_channels.to_le_bytes());
    let _ = w.write_all(&sfx.waveformatex.n_samples_per_sec.to_le_bytes());
    let _ = w.write_all(&sfx.waveformatex.n_avg_bytes_per_sec.to_le_bytes());
    let _ = w.write_all(&sfx.waveformatex.n_block_align.to_le_bytes());
    let _ = w.write_all(&sfx.waveformatex.w_bits_per_sample.to_le_bytes());
    let _ = w.write_all("data".as_bytes());
    let _ = w.write_all(&(sfx.waveformdata.len() as u32).to_le_bytes());
    let _ = w.write_all(&sfx.waveformdata);
}

#[derive(Debug)]
pub enum SharedResource {
    Null,
    //Byte(u8),
    //SByte(i8),
    //Int16(i16),
    //UInt16(u16),
    //Int32(i32),
    //UInt32(u32),
    //Int64(i64),
    //UInt64(u64),
    //Single(f32),
    //Double(f64),
    //Boolean(bool),
    //Char(char),
    //String(String),
    //Array(Vec<Resource>),
    //List(Vec<Resource>),
    //Dictionary(BTreeMap<Resource, Resource>),
    SoundEffect(SoundEffect),
    Unsupported(String),
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
    
    match type_readers.get(reader_id as usize)
        .ok_or(XnbError::InvalidTypeReaderId(reader_id))?.0.as_str() {
        "Microsoft.Xna.Framework.Content.SoundEffectReader" => {
            read_sound_effect(r)
        },
        name => {
            Ok(SharedResource::Unsupported(name.to_string()))
        },
    }
}

pub struct SoundEffect {
    waveformatex: WaveFormatEx,
    waveformdata: Vec<u8>,
    loop_start: i32,
    loop_len: i32,
    duration_ms: i32,
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

// https://learn.microsoft.com/en-us/windows/win32/api/mmeapi/ns-mmeapi-waveformatex
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

fn read_dotnet_string(r: &mut dyn Read) -> Result<String, XnbError> {
    let string_len = read_seven_bit_encoded_int(r)?;
    let string_data = read_until_len_or_null(r, string_len as usize)?;
    String::from_utf8(string_data).map_err(XnbError::Utf8)
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

fn read_u8(r: &mut dyn Read) -> Result<u8, XnbError> {
    let mut num_data = [0u8; 1];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(num_data[0])
}

fn read_u16(r: &mut dyn Read) -> Result<u16, XnbError> {
    let mut num_data = [0u8; 2];
    r.read_exact(&mut num_data).map_err(XnbError::Io)?;
    Ok(u16::from_le_bytes(num_data))
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
    Invalid7BitInt(u8),
    InvalidFileFormat([u8; 3]),
    InvalidPlatform(u8),
    InvalidTypeReaderId(i32),
    InvalidFormatSize(u32),
    InvalidDataSize(u32),
    UnimplementedVersion(u8),
    UnimplementedPlatform(TargetPlatform),
    NoTypeReaders,
    NoSharedResources,
    UnimplementedWavCodec(u16),
}

impl fmt::Display for XnbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for XnbError { }

