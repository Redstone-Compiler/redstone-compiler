
module not_clk(clk, clk_n);
  input clk;
  output clk_n;
  assign clk_n = ~clk;
endmodule

module d_latch(d, en, q);
  input d, en;
  output reg q;
  always @(*) begin
    if (en) begin
      q <= d;
    end
  end
endmodule

module d_flip_flop(d, clk, q);
  input d, clk;
  output q;
  wire clk_n, master_q;
  not_clk inv(.clk(clk), .clk_n(clk_n));
  d_latch master(.d(d), .en(clk_n), .q(master_q));
  d_latch slave(.d(master_q), .en(clk), .q(q));
endmodule
