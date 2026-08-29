use neoethos_search::resident_search_slice2_v3::{ResidentSearchArchiveStagedV3, ResidentSearchRankEnqueuedV3};

fn ranked() -> ResidentSearchRankEnqueuedV3 {
    loop {}
}

fn main() {
    let _ = ResidentSearchArchiveStagedV3::from_ranked_v3(ranked());
}
