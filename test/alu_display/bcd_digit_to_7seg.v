// 첫 물리 표시 실험: 세 자리 000~255 표시의 기본 모드인 blank=0.
// bcd_to_7seg.v 또는 bcd_to_7seg_sop.v와 함께 컴파일한다.
module bcd_digit_to_7seg(digit, seg);
  input [3:0] digit;
  output [6:0] seg;
  bcd_to_7seg decoder(.digit(digit), .blank(1'b0), .seg(seg));
endmodule
