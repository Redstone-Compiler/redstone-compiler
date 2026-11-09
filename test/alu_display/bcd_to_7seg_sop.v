// 공유 숫자 판정과 segment OR. bit0=a, active-high, invalid/blank=OFF.
module bcd_to_7seg(digit, blank, seg);
  input [3:0] digit;
  input blank;
  output [6:0] seg;
  wire d0, d1, d2, d3, d4, d5, d6, d7, d8, d9;
  wire a, b, c, d, e, f, g;
  assign d0 = digit == 4'd0;
  assign d1 = digit == 4'd1;
  assign d2 = digit == 4'd2;
  assign d3 = digit == 4'd3;
  assign d4 = digit == 4'd4;
  assign d5 = digit == 4'd5;
  assign d6 = digit == 4'd6;
  assign d7 = digit == 4'd7;
  assign d8 = digit == 4'd8;
  assign d9 = digit == 4'd9;
  assign a = ~blank & (d0 | d2 | d3 | d5 | d6 | d7 | d8 | d9);
  assign b = ~blank & (d0 | d1 | d2 | d3 | d4 | d7 | d8 | d9);
  assign c = ~blank & (d0 | d1 | d3 | d4 | d5 | d6 | d7 | d8 | d9);
  assign d = ~blank & (d0 | d2 | d3 | d5 | d6 | d8 | d9);
  assign e = ~blank & (d0 | d2 | d6 | d8);
  assign f = ~blank & (d0 | d4 | d5 | d6 | d8 | d9);
  assign g = ~blank & (d2 | d3 | d4 | d5 | d6 | d8 | d9);
  assign seg = {g, f, e, d, c, b, a};
endmodule
