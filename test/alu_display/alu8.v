// op: 0 ADD, 1 SUB, 2 AND, 3 OR, 4 XOR, 5 PASS A, 6 SHL1, 7 SHR1.
// C: ADD carry / SUB no-borrow / shift에서 빠진 bit. V: ADD/SUB signed overflow.
module alu8(a, b, op, y, z, n, c, v);
  input [7:0] a, b;
  input [2:0] op;
  output [7:0] y;
  output z, n, c, v;
  wire sub;
  wire [7:0] rhs;
  wire [8:0] wide;
  assign sub = op == 3'd1;
  assign rhs = sub ? ~b : b;
  assign wide = {1'b0, a} + {1'b0, rhs} + (sub ? 9'd1 : 9'd0);
  assign y = op == 3'd0 ? wide[7:0] :
             op == 3'd1 ? wide[7:0] :
             op == 3'd2 ? a & b :
             op == 3'd3 ? a | b :
             op == 3'd4 ? a ^ b :
             op == 3'd5 ? a :
             op == 3'd6 ? {a[6:0], 1'b0} : {1'b0, a[7:1]};
  assign z = y == 8'd0;
  assign n = y[7];
  assign c = op == 3'd0 ? wide[8] :
             op == 3'd1 ? wide[8] :
             op == 3'd6 ? a[7] :
             op == 3'd7 ? a[0] : 1'b0;
  assign v = op == 3'd0 ? (~(a[7] ^ b[7]) & (y[7] ^ a[7])) :
             op == 3'd1 ? ((a[7] ^ b[7]) & (y[7] ^ a[7])) : 1'b0;
endmodule
