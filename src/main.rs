use std::fs;
use std::path::{Path, PathBuf};

use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};

use rayon::prelude::*;

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
struct Completion {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
    system_fingerprint: String,
    x_groq: Groq,
}

#[derive(Serialize, Deserialize, Debug)]
struct Choice {
    index: u8,
    message: Message,
    logprobs: Option<HashMap<String, f64>>,
    finish_reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Usage {
    prompt_tokens: u32,
    prompt_time: f64,
    completion_tokens: u32,
    completion_time: f64,
    total_tokens: u32,
    total_time: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Groq {
    id: String,
}

fn convert_pdf_to_png(pdf_path: &str, output_path: &str) -> std::io::Result<()> {
    // Construct the command
    // For ImageMagick version 7 and newer, use `magick convert` instead of `convert`
    //let output = format!("{}/{}", output_path, "page-%04d.png");
    let output = format!("{}/{}", output_path, "page");

    // fast, but depending on pdf, multiple images per page
    let status = std::process::Command::new("pdfimages")
        .arg(pdf_path)
        .arg(output)
        .status()?;

    // takes a long time but gives one png per page
    //let status = std::process::Command::new("convert")
    //    .arg("-density")
    //    .arg("150")
    //    .arg(pdf_path)
    //    .arg("-quality")
    //    .arg("90")
    //    .arg(output)
    //    .status()?;

    if status.success() {
        println!("PDF conversion successful.");
    } else {
        eprintln!("PDF conversion failed.");
    }

    Ok(())
}

fn list_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            //if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("png") {
            if path.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn ocr(filename: &str, language: &str) -> Result<String, tesseract::TesseractError> {
    let mut t = tesseract::Tesseract::new(None, Some(language))?;
    t.set_page_seg_mode(tesseract::PageSegMode::PsmAuto);
    Ok(t
        .set_image(filename)?
        .recognize()?
        .get_text()?)
}

fn has_text(image_path: &str) -> Option<(image::DynamicImage, image::GrayImage)> {
    // Load the image
    let image = image::open(image_path).unwrap();
    let gray_image = image.to_luma8();

    // Perform Canny edge detection
    let edges = imageproc::edges::canny(&gray_image, 50.0, 100.0);

    // Count the number of distinct edges
    let edge_count = count_edges(&edges);

    // Threshold for determining if the image contains text
    let threshold = 50_000; // Adjust this based on your needs

    println!("{}: {}", image_path, edge_count);
    if edge_count > threshold {
        Some((image, edges))
    } else {
        None
    }
}

fn count_edges(image: &image::GrayImage) -> u32 {
    image.pixels().filter(|&p| p[0] > 0).count() as u32
}

//async fn correct_with_llm(text: &str) -> anyhow::Result<String> {
async fn correct_with_groq(text: &str) -> anyhow::Result<Completion> {
    let api_key = std::env::var("GROQ_API_KEY").expect("API key not set");

    // Setting up the headers
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, format!("Bearer {}", api_key).parse().unwrap());
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());

    // Creating the client
    let client = reqwest::Client::new();

    // JSON body as per your example
    let body = serde_json::json!({
      "messages": [
        {
          "role": "system",
          "content": "You are an expert editor. The user will give you one or more scanned pages to correct. Correct any errors you find, but at a minimum you must:\n1. Fix spelling mistakes\n2. Remove page headers, footers, and footnotes\n3. Remove in-text footnote references, which may appear as random, out-of-place characters\n4. Combine sentences separated by newlines into one line. Combine sentences in the same paragraph into one line\n5. Separate into paragraphs\nEcho the text exactly as is except for corrections. Don't print anything except the corrected text."
        },
        {
          "role": "user",
          "content": text,
        },
      ],
      "model": "llama3-70b-8192",
      "temperature": 1,
      "max_tokens": 8192,
      "top_p": 1,
      "stream": false,
      "stop": null
    });

    // Making the POST request
    let res = client.post("https://api.groq.com/openai/v1/chat/completions")
        .headers(headers)
        .json(&body)
        .send()
        .await?;

    if res.status() != reqwest::StatusCode::OK {
        return Err(anyhow::anyhow!(
            "status not OK: {}; {}",
            res.status(),
            res.text().await.unwrap_or_default(),
        ));
    }
    Ok(serde_json::from_str(&res.text().await?)?)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pdf_name = "octaviusofminuci00minuiala.pdf";
    let processing_dir_name = "/tmp/octavius";
    let processing_dir = Path::new(processing_dir_name);
    let txt_name = "octavius.txt";
    std::fs::create_dir_all(processing_dir_name).unwrap();

    let start = std::time::Instant::now();
    println!("Converting PDF to pngs...");
    convert_pdf_to_png(pdf_name, processing_dir_name).unwrap();
    println!("Done converting PDF to pngs in {:?}.", start.elapsed());

    let files = list_files(&processing_dir);

    // ~68s not parallelized
    // ~20s parallelized
    let start = std::time::Instant::now();
    println!("OCRing pages...");
    let pages = files
        .par_iter()
        .filter_map(|file| {
            let path = file.to_string_lossy().to_string();
            has_text(&path)
                .map(|_| ocr(&path, "eng").unwrap_or_default())
        })
        .collect::<Vec<String>>();
    println!("Done OCRing pages in {:?}.", start.elapsed());

    println!("\n\nUncorrected text:");
    //for page in &pages {
    for page in &pages[1..3] {
        println!("{}", page);
        println!("");
    }

    let mut corrected_pages = vec![];
    //for page in &pages {
    for page in &pages[1..3] {
        corrected_pages.push(correct_with_groq(page).await?);
    }

    println!("\n\nCorrected text:");
    for page in corrected_pages {
        for choice in page.choices {
            println!("{}", choice.message.content);
            println!("");
        }
    }

    //let text = texts
    //    .iter()
    //    .fold(String::new(), |mut acc, ocr| {
    //        acc.push_str(&ocr);
    //        acc
    //    });

    //std::fs::write(out, &text).unwrap();

    Ok(())
}
