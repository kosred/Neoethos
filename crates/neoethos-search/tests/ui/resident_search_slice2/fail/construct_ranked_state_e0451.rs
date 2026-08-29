use neoethos_search::resident_search_slice2_v3::ResidentSearchRankEnqueuedV3;

fn ranked() -> ResidentSearchRankEnqueuedV3 {
    loop {}
}

fn main() {
    let _ = ResidentSearchRankEnqueuedV3 { ..ranked() };
}
