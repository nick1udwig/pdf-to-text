# pdf-to-text

A simple Rust program to:
1. OCR a pdf, page-by-page,
2. Correct common OCR errors & format text using an LLM (here, Llama3-70B).

The goal is to eventually:
1. Expose this as a service over the [Kinode](https://github.com/kinode-dao/kinode) network,
2. Develop an even simpler TTS service & expose that over the Kinode network as well.
Then, users will be able to submit public-domain PDFs to these two services and, for less than a dollar, get back a text copy, suitable for an ereader, and an audiobook.
