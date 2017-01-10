% LLE algorithm( using K nearest neighbors)
% [Y]=lle(X,K,dmax)
% X=data as D*N matrix (D=dimensionality,N= number)
% K=the number of neighbors
% dmax=max embedding dimensionality
% Y=embedding as dmax*N matrix
function [Y]=lle(X,K,d);
[D,N]=size(X);
fprintf('LLE running on %d points in %d dimensions\n',N,D);
% Step1: compute pairwise distances and find neighbors
fprintf('Finding %d nearest neighbors.\n',K);
X2=sum(X.^2,1);
distance=repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
[sorted,index]=sort(distance);
neighborhood=index(2:(1+K),:);
% Step2:solve for recnstruction weights
fprintf('Solving for reconstruction weights.\n');
if(K>D)
    fprintf('[note:K>D;regularization will be used]\n');
    tol=1e-3;
else
    tol=0;
end
W=zeros(K,N);
for ii=1:N
    z=X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K);%shift i th pt to origin
    C=z'*z;% local covariance
    C=C+eye(K,K)*tol*trace(C);%regularlization(K>D)
    W(:,ii)=C\ones(K,1); %b\a 相当于b除a，或者是(b的倒数)乘以a
    W(:,ii)=W(:,ii)/sum(W(:,ii));% enforce sum(W)=1
end
% other possible regularizes for K>D
% C=C+tol*diag(diag(C)); %regularlization
% C=C+eye(K,K)*tol*trace(C)*K; %regularlization


% Step3:compute embedding from eigenvects of cost matrix M=(I-W)'*(I-W)
fprintf('----------computing embedding----------');
% using a sparse matrix with storage for 4KN nonzero elements
M=sparse(1:N,1:N,ones(1,N),N,N,4*K*N);
for ii=1:N
    w=W(:,ii);
    jj=neighborhood(:,ii);
    M(ii,jj)=M(ii,jj)-w';
    M(jj,ii)=M(jj,ii)-w;
    M(jj,jj)=M(jj,jj)+w*w';
end
% Calculation of embedding
options.disp=0;
options.isreal=1;
options.issym=1;
[Y,eigenvals]=eigs(M,d+1,0,options);% eigs is not robust
Y=Y(:,2:d+1)'*sqrt(N);% bottom evect is [1,1,1,1.....] with eval 0
fprintf('Done.\n');


    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    