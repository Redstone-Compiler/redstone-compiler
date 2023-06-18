module full_addder (
  input x,
  input y,
  input c_in,
  output s,
  output c_out,
);

  assign s = (x ^ y) ^ c_in
  assign c_out = (a & b) | (s & c_in)

endmodule