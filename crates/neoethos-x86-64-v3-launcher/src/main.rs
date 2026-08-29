use std::process;

use neoethos_x86_64_v3_launcher::run_public_launcher_v1;

fn main() {
    match run_public_launcher_v1() {
        Ok(payload_exit_code) => process::exit(payload_exit_code),
        Err(error) => {
            eprintln!("{error}");
            process::exit(error.exit_code());
        }
    }
}
