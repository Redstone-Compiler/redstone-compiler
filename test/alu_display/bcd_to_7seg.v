// seg[0..6] = a,b,c,d,e,f,g. 1이면 해당 segment를 켠다.
// blank 우선, BCD 범위 밖(10..15)은 모두 끈다.
module bcd_to_7seg(digit, blank, seg);
  input [3:0] digit;
  input blank;
  output [6:0] seg;

  assign seg = blank ? 7'h00 :
    digit == 4'd0 ? 7'h3f :
    digit == 4'd1 ? 7'h06 :
    digit == 4'd2 ? 7'h5b :
    digit == 4'd3 ? 7'h4f :
    digit == 4'd4 ? 7'h66 :
    digit == 4'd5 ? 7'h6d :
    digit == 4'd6 ? 7'h7d :
    digit == 4'd7 ? 7'h07 :
    digit == 4'd8 ? 7'h7f :
    digit == 4'd9 ? 7'h6f : 7'h00;
endmodule
