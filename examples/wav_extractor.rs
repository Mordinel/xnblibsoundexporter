use std::fs::{self, File};
use std::io::{Read, BufReader};

use xnbparse::{SharedResource, XnbData, XnbError};

fn main() -> Result<(), std::io::Error> {
    let args = std::env::args().collect::<Vec<String>>();    
    if args.len() != 2 {
        print_usage(&args[0]);
        return Ok(());
    }

    let file = File::open(&args[1])?;
    let mut reader = BufReader::new(file);

    let data: Result<XnbData, XnbError> = (&mut reader as &mut dyn Read).try_into();
    if let Err(e) = data {
        eprintln!("ERROR: {e}");
        return Ok(());
    }

    let data = data.unwrap();
    data.shared_resources.into_iter()
        .filter_map(|p| match p {
            SharedResource::SoundEffect(s_e) => Some(s_e.file_data().ok()?),
            _ => None,
        })
        .enumerate()
        .for_each(|(i, effect)| {
            let filename = format!("0{i}.wav");
            let _ = fs::write(filename.clone(), &effect).ok();
            println!("Extracted '{filename}'");
        });

    return Ok(());
}

fn print_usage(executable: &str) {
    println!("Usage: {executable} <filename>.xnb");
}

