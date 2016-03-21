
clear;
clc;
close all;

load ../data/cvx.h5;
cx = data';
load ../data/cvy.h5;
cy = data';
load ../data/dvx.h5;
dx = data';
load ../data/dvy.h5;
dy = data';
load ../data/dxu.h5;
xx = data';
load ../data/dyu.h5;
yy = data';

w=640;
h=640;
n=64;

i=590+1;
j=22+1;



xx(j,i)
yy(j,i)

pdx = dx(64*((j-1)*w+i-1)+1:64*((j-1)*w+i-1)+1+n-1,1);
pdy = dy(64*((j-1)*w+i-1)+1:64*((j-1)*w+i-1)+1+n-1,1);
pcx = cx(64*((j-1)*w+i-1)+1:64*((j-1)*w+i-1)+1+n-1,1)./64;
pcy = cy(64*((j-1)*w+i-1)+1:64*((j-1)*w+i-1)+1+n-1,1)./64;


hold on;
plot(pcx,'or');
figure;
plot(pcy,'ob');
figure;
plot(pdx,'xr');
figure;
plot(pdy,'xb');