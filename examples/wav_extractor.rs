use std::{fs::File, io::{BufReader, Read}};

use xnbsoundexporter::{XnbData, XnbError};

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
    println!("{data:#?}");

    return Ok(());
}

fn print_usage(executable: &str) {
    println!("Usage: {executable} <filename>.xnb");
}

