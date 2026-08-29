use neoethos_search::resident_search_slice2_v3::ResidentSearchGenerationChainV3;

fn chain() -> ResidentSearchGenerationChainV3 {
    loop {}
}

fn require_copy<T: Copy>(_: T) {}

fn main() {
    require_copy(chain());
}
