module full_adder(a, b, cin, sum, cout);
  input a, b, cin;
  output sum, cout;
  wire carry_generate, partial_sum, carry_propagate;

  assign carry_generate = a & b;
  assign partial_sum = a ^ b;
  assign carry_propagate = partial_sum & cin;
  assign sum = partial_sum ^ cin;
  assign cout = carry_generate | carry_propagate;
endmodule
