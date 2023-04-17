def forward_export(self, x):
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        x[i] = x[i].sigmoid() 
    return x 
