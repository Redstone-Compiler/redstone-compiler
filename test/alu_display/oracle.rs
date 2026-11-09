// RTL의 hex 표와 별도로, 켜져야 할 segment의 이름으로 기대값을 정의한다.
pub fn segment_mask(digit: usize, blank: bool) -> usize {
    if blank || digit > 9 {
        return 0;
    }
    let lit = [
        "abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg",
    ][digit];
    lit.bytes()
        .fold(0, |mask, segment| mask | (1 << (segment - b'a')))
}
