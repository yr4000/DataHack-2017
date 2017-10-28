%% Amplitude spectrum with fft
%if n=0 the function takes the length of the whole trace
%if norm_flag=1 the function gives normalized power (normalized to NFFT/2)
function [Y_abs,f] = fn_fftAmp(y,sf,n,norm_flag)

L = size(y,1);
if nargin >= 3 && n~=0;
    NFFT = 2^nextpow2(n); % Next power of 2 from length of n;
else
    NFFT = 2^nextpow2(L); % Next power of 2 from length of y
end
Y = fft(y,NFFT)/L;
f = sf/2*linspace(0,1,NFFT/2+1);
Y_abs= (abs(Y(1:NFFT/2+1,:))).^2;
if norm_flag==1
    Y_abs=Y_abs./(NFFT/2);
end
Y_plot = mean(Y_abs,2);

% % Plot single-sided amplitude spectrum.
% figure
% plot(f,Y_plot) 
% title('Single-Sided Amplitude Spectrum of y(t)')
% xlabel('Frequency [Hz]')
% ylabel('Power spectral density [mV^2/Hz]')