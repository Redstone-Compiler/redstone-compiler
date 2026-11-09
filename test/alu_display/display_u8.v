// 000..255를 세 자리로 직접 표시한다. seg[0..6]은 a..g, active-high.
module display_u8(value, hundreds_seg, tens_seg, ones_seg);
  input [7:0] value;
  output [6:0] hundreds_seg, tens_seg, ones_seg;
  wire [3:0] hundreds, tens, ones;
  bin8_to_bcd convert(.value(value), .hundreds(hundreds), .tens(tens), .ones(ones));
  bcd_to_7seg h(.digit(hundreds), .blank(1'b0), .seg(hundreds_seg));
  bcd_to_7seg t(.digit(tens), .blank(1'b0), .seg(tens_seg));
  bcd_to_7seg o(.digit(ones), .blank(1'b0), .seg(ones_seg));
endmodule
