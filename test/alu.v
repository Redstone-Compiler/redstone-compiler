module fulladdder (a, b, cin, sum, cout);
  input a, b, cin;
  output sum, cout;

  s = (a ^ b) ^ cin;
  cout = (a & b) | (s & cin);
endmodule
