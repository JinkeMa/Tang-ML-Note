

%%
I=im2double(((imread('c:/users/dell/desktop/lena.png'))));I=imresize(I,[512 512]);
gray=I;
sigma=35;
I=imnoise(I,'gaussian',0,sigma^2/255^2);
[h,w]=size(I);
%%
k_hash=[1,-1];
out=3;in=2;
g=grad(I);
g=mat2gray(g);

I_padded=padarray(I,[out+in out+in],'symmetric');%边界作对称处理
% psnr_hash=zeros(100,3);
psnr_hash=zeros(61,11);
result_pic=cell(100,1);
%%
psnr_hash(2:61,1)=(1:60)/500;
psnr_hash(1,2:11)=(1:10)/2;
%寻找power1、power2与sigma的关系，power范围自己设置，以减少计算次数为目标。
for power2=44:48
    for power1=1:1
        gama1hash=imfilter(g,k_hash,'replicate');
        gama1hash(gama1hash<=0)=0;
        gama1hash(gama1hash>0)=1;
       %%
        gama1_padded=padarray(gama1hash,[out+in out+in],'symmetric');%边界做对称处理
        O_padded=padarray(I,[out out],'symmetric');%边界作对称处理
        weight_sum_c=zeros(h,w);
        result_sum_c=weight_sum_c;
        wmax=weight_sum_c;
        wmax2=weight_sum_c;
       %%
        for offset1=-out:out
            for offset2=-out:out
                if offset1==0&&offset2==0
                    continue
                end
                [weight_conv,weight_hash1]=integral_img3(I_padded,gama1_padded,out,offset1,offset2,in);
                now=O_padded(offset1+out+1:offset1+out+h,offset2+out+1:offset2+out+w);
                %%
                %反映的是两个距离权值所占的比重的问题。欧式距离可以衡量主要的相似度，所以占的比重更大，汉明距离占的比重应该更小才对
                %其次，引入汉明距离目的是对权值进行微调，而不是让其(汉明权值)占主导地位。观察高斯权值函数的特点：随着自变量增大，梯度越来越小，
                %也就是说，自变量在一个很大的范围内时，因变量被压缩的很厉害。所以考虑增大汉明距离值的范围，以达到此目的（这样，汉明权值就
                %被压缩在一个很小的范围，不同的汉明权值之间的差异很小，从而达到微调的目的）。本文实现方法是对欧式距离和汉明距离采用不同的卷积
                %算子，欧式距离采用的卷积算子为ones(2*in+1,2*in+1)/(2*in+1)^2,h汉明距离采用的为ones(2*in+1,2*in+1)，体现在下面的integral_img3()函数里
                
                %%
                %设置power1和power2，，并观察不同方差的噪声图像其最好的psnr值对应的power1和power2有怎样的变化。
                weight_=weight_conv.*weight_hash1.^(power1/2);
                weight_=exp(-weight_/(power2/500));
                wmax=max(wmax,weight_);
                result_sum_c=result_sum_c+weight_.*now;
                weight_sum_c=weight_sum_c+weight_;
            end
        end
        center=O_padded(out+1:out+h,out+1:out+w);
        result_sum_c=result_sum_c+wmax.*center;
        weight_sum_c=weight_sum_c+wmax;
        result=result_sum_c./(weight_sum_c);
%         result_pic{(power1-1)*10+power2,:}=result;
%         psnr_hash((power1-1)*10+power2,:)=[power1/50,power2/50,psnr(result,gray)];
        psnr_hash(power2+1,power1+1)=psnr(result,gray);
%         psnr_hash=psnr(result,gray);
    end
end
%%
%
function [weight_conv,weight_hash1] = integral_img3(I_padded,gama1_padded,out,offset1,offset2,in)
    o_dist=(I_padded(1+out:end-out,1+out:end-out)-I_padded(1+out+offset1:end-out+offset1,1+out+offset2:end-out+offset2)).^2;
    xor_1=xor(gama1_padded(1+out:end-out,1+out:end-out),gama1_padded(1+out+offset1:end-out+offset1,1+out+offset2:end-out+offset2));
    kernel1=ones(2*in+1,2*in+1)/(2*in+1)^2;
    kernel2=ones(2*in+1,2*in+1);
%     kernel3=ones(2*in+1,2*in+1)/(2*in+1)^2;
    weight_conv=conv2(o_dist,kernel1,'valid');
    weight_hash1=conv2(xor_1,kernel2,'valid');
%     weight_hash2=conv2(xor_1,kernel3,'valid');
end
%求梯度
function [g]=grad(I)
    blur=imgaussfilt(I,1.2);
    k_sobel1=[-1 -2 -1;0 0 0;1 2 1];
    k_sobel2=[-1 0 1;-2 0 2;-1 0 1];
    g1=imfilter(blur,k_sobel1,'replicate');
    g2=imfilter(blur,k_sobel2,'replicate');
    g=abs(g1)+abs(g2);
end
