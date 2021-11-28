#!/bin/bash

RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --no-default-features --target wasm32-unknown-unknown
wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/debug/mdlb.wasm
cp src/index.html target/generated/index.html

echo "Build done ! (or not)"
echo "you can now run a web server in target/generated"
echo "ex. python3 -m http.server --directory target/generated"
echo "and go http://localhost:8000/index.html"
echo "(use correct port)"
