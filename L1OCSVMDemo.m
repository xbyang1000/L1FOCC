function L1OCSVMDemo()
clear;clc;
load dataX;
X1 = X;
X = X1(1:200,:);% take the first 100 samples
n = size(X,1); 
espsilon = 0.001;

linewidth = 2;
%% L1-OCSVM
[C,sigma,alpha,rho,Corr,Compactness] = ParaTuning(X,X1(201:end,:));
%shift = 1; str = 'g-.';
%C = 1000; sigma = 3.4; % Need to be tuning
figure(1); hold on;

a = ([-10 15 -15 25]);
ind = find(abs(alpha)>= espsilon);  % SV
SV = X(ind,:);
fprintf("\n SV in L1OSVM %d\n",size(ind,1));
p1 = show_contour(X,alpha,'RBF',sigma,rho,'g-',linewidth+8,1,a);


[C,sigma,alpha,rho,Corr,Compactness] = ParaTuning([X;Outliers(1,:)],X1(201:end,:));
figure(1);hold on;
p2 = show_contour([X;Outliers(1,:)],alpha,'RBF',sigma,rho,'y-.',linewidth+5,1,a);

% [C,sigma,alpha,rho,Corr,Compactness] = ParaTuning([X;Outliers(1:2,:)],X1(201:end,:));
% figure(1);hold on;
% show_contour([X;Outliers(1:2,:)],alpha,'RBF',sigma,rho,'y-.',linewidth+3,1,a);

[C,sigma,alpha,rho,Corr,Compactness] = ParaTuning([X;Outliers(1:3,:)],X1(201:end,:));
figure(1);hold on;
p3 = show_contour([X;Outliers(1:3,:)],alpha,'RBF',sigma,rho,'r:',linewidth+1,1,a);

% [C,sigma,alpha,rho,Corr,Compactness] = ParaTuning([X;Outliers(1:4,:)],X1(201:end,:));
% figure(1);hold on;
% show_contour([X;Outliers(1:4,:)],alpha,'RBF',sigma,rho,'r--',linewidth+1,1,a);

[C,sigma,alpha,rho,Corr,Compactness] = ParaTuning([X;Outliers(1:5,:)],X1(201:end,:));
figure(1);hold on;
p4 = show_contour([X;Outliers(1:5,:)],alpha,'RBF',sigma,rho,'k-',linewidth-1,1,a);


hold on; box on;grid on;
p5 = plot(X(:,1),X(:,2),'b.','Markersize',8);hold on
p6 = plot(X1(201:end,1),X1(201:end,2),'c.','Markersize',8);
%legend('Support vectors','0-outlier plane','1-outlier plane','3-outlier plane','5-outlier plane','Training data','Testing data','Outliers');
hold on;
p7 = plot(Outliers(:,1),Outliers(:,2),'r+','linewidth',2,'Markersize',8);
text(Outliers(:,1),Outliers(:,2)-2,num2str((1:1:5)'),'FontSize',13,'fontweight','bold','color','b');
hold on; 
p8 = plot(SV(:,1),SV(:,2),'mo','LineWidth',2, 'Markersize',8);
legend({'0-outlier plane','1-outlier plane','3-outlier plane','5-outlier plane','Training data','Testing data','Outliers','Support vectors'});
xlabel('x');
ylabel('y');
xticks([-30:10:20]);yticks([-15:10:25]);

set(gcf,'color','w');
set(gca,'fontweight','bold','FontSize',13,'fontname','Times new roman');
axis([-30 20 -20 25]);




%% OCSVM

figure(2);hold on
[C,sigma,alpha,rho,Corr,Compactness] = ParaTuningOCSVM(X,X1(201:end,:),espsilon);
alpha1 = alpha;

a = ([-10 15 -15 25]);
p1 = show_contour(X,alpha,'RBF',sigma,rho,'g-',linewidth+8,20,a);


[C,sigma,alpha,rho,Corr,Compactness] = ParaTuningOCSVM([X;Outliers(1,:)],X1(201:end,:),espsilon);
figure(2);hold on;
p2 = show_contour([X;Outliers(1,:)],alpha,'RBF',sigma,rho,'y-.',linewidth+5,20,a);


[C,sigma,alpha,rho,Corr,Compactness] = ParaTuningOCSVM([X;Outliers(1:3,:)],X1(201:end,:),espsilon);
figure(2);hold on;
p3 = show_contour([X;Outliers(1:3,:)],alpha,'RBF',sigma,rho,'r:',linewidth+1,20,a);


[C,sigma,alpha,rho,Corr,Compactness] = ParaTuningOCSVM([X;Outliers(1:5,:)],X1(201:end,:),espsilon);
figure(2);hold on;
p4 = show_contour([X;Outliers(1:5,:)],alpha,'RBF',sigma,rho,'k-',linewidth-1,30,a);


hold on; box on;grid on;
p5 = plot(X(:,1),X(:,2),'b.','Markersize',8);hold on
p6 = plot(X1(201:end,1),X1(201:end,2),'c.','Markersize',8);

hold on;
p7 = plot(Outliers(:,1),Outliers(:,2),'r+','linewidth',2,'Markersize',8);
text(Outliers(:,1),Outliers(:,2)-2,num2str((1:1:5)'),'FontSize',13,'fontweight','bold','color','b');
hold on; 
%LegendLabel = [{'0-outlier plane';'1-outlier plane';'3-outlier plane';'5-outlier plane';'Training data';'Testing data';'Outliers';'Support vectors'}];
%ind = (alpha1>1e-11);
SV = X;
p8 = plot(SV(:,1),SV(:,2),'mo','LineWidth',2, 'Markersize',8);

xlabel('x');
ylabel('y');
xticks([-30:10:20]);yticks([-15:10:25]);

set(gcf,'color','w');
set(gca,'fontweight','bold','FontSize',13,'fontname','Times new roman');
axis([-40 30 -30 35]);
hold off;

%% SVDD 
i = 3;
figure(i);hold on
[C,sigma,alpha,RR,Corr,Compactness] = ParaTuningSVDD(X,X1(201:end,:),espsilon);
% rho -> RR
alpha1 = alpha;

a = ([-10 15 -15 25]);
p1 = show_contourSVDD(X,alpha,'RBF',sigma,RR,'g-',linewidth+8,20,a);


[C,sigma,alpha,RR,Corr,Compactness] = ParaTuningSVDD([X;Outliers(1,:)],X1(201:end,:),espsilon);
figure(i);hold on;
p2 = show_contourSVDD([X;Outliers(1,:)],alpha,'RBF',sigma,RR,'y-.',linewidth+5,20,a);


[C,sigma,alpha,RR,Corr,Compactness] = ParaTuningSVDD([X;Outliers(1:3,:)],X1(201:end,:),espsilon);
figure(i);hold on;
p3 = show_contourSVDD([X;Outliers(1:3,:)],alpha,'RBF',sigma,RR,'r:',linewidth+1,20,a);


[C,sigma,alpha,RR,Corr,Compactness] = ParaTuningSVDD([X;Outliers(1:5,:)],X1(201:end,:),espsilon);
figure(i);hold on;
p4 = show_contourSVDD([X;Outliers(1:5,:)],alpha,'RBF',sigma,RR,'k-',linewidth-1,20,a);


hold on; box on;grid on;
p5 = plot(X(:,1),X(:,2),'b.','Markersize',8);hold on
p6 = plot(X1(201:end,1),X1(201:end,2),'c.','Markersize',8);

hold on;
p7 = plot(Outliers(:,1),Outliers(:,2),'r+','linewidth',2,'Markersize',8);
text(Outliers(:,1),Outliers(:,2)-2,num2str((1:1:5)'),'FontSize',13,'fontweight','bold','color','b');
hold on; 
%LegendLabel = [{'0-outlier plane';'1-outlier plane';'3-outlier plane';'5-outlier plane';'Training data';'Testing data';'Outliers';'Support vectors'}];
ind = (alpha1>espsilon);
SV = X(ind,:);
p8 = plot(SV(:,1),SV(:,2),'mo','LineWidth',2, 'Markersize',8);

xlabel('x');
ylabel('y');
xticks([-30:10:20]);yticks([-15:10:25]);

set(gcf,'color','w');
set(gca,'fontweight','bold','FontSize',13,'fontname','Times new roman');
axis([-30 20 -20 25]);
hold off;


end


function [C0,sigma0,alpha0,rho0,Corr,Compactness] = ParaTuningSVDD(TrX,TeX,epsilon)
% gamma = 1/(2*sigma^2);
n = size(TrX,1); 
X = TrX;
options = optimoptions(@quadprog,'Algorithm','interior-point-convex','Display', 'none');
Corr = [0;0]; Compactness = 1e10;
for j = -2:2
        gamma = 10^j; sigma = sqrt(gamma/2);
        %sigma = 15;
    for i = -3:3
        nv = 10^i; C = 1/(nv*n);   
        %C = 10^i;
        K = RBFkernel(X,X,sigma,1);
        H = K./2;
        f = -diag(K);
        A = [eye(n);-eye(n)];
        b = [ones(n,1)*C;zeros(n,1)];
        Aeq = ones(1,n);
        beq = 1;
        %x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)
        solution =quadprog(H,f,A,b,Aeq,beq,[],[],[],options);
        if (isempty(solution)), continue;end
        alpha = solution;
        
        
        ind = (alpha>epsilon & alpha<C); % Get SV
        if(isempty(ind)),continue;end
        Tmp = 2*K(ind,:)*alpha;
        RR = 1 + alpha'*K*alpha - sum(Tmp)/n;  % Get RR:, i.e., R^2
        % RR > ||\phi(x)- sum_i (alpha_i *\phi(x_i) )|| 
        TrainResult = RR - 1 - alpha'*K*alpha + 2*K*alpha;         
        TrainCorr = 100*sum(TrainResult>=0)/n;
        if (TrainCorr>=Corr(1))
            Corr(1) = TrainCorr; C0 = C;sigma0 = sigma;
            if (min(TrainResult) < Compactness)
                Compactness = min(TrainResult);
                C0 = C;sigma0 = sigma;
                alpha0 = alpha;rho0 = RR;
            end            
        end 
    end
end

   K_Te = RBFkernel(TeX,X,sigma0,2);
   TestResult = RR - 1 - alpha'*K*alpha + 2*K_Te*alpha;
   Corr(2) = 100*sum(TestResult>=0)/size(TeX,1);
end




function [C0,sigma0,alpha0,rho0,Corr,Compactness] = ParaTuningOCSVM(TrX,TeX,epsilon)
% gamma = 1/(2*sigma^2);
n = size(TrX,1); 
X = TrX;
options = optimoptions(@quadprog,'Algorithm','interior-point-convex','Display', 'none');
Corr = [0;0]; Compactness = 1e10;
for j = -2:2
        gamma = 10^j; sigma = sqrt(gamma/2);
        %sigma = 15;
    for i = -3:3
        nv = 10^i; C = 1/(nv*n);   
        %C = 10^i;
        K = RBFkernel(X,X,sigma,1);
        H = K./2;
        f = zeros(n,1);
        A = [eye(n);-eye(n)];
        b = [ones(n,1)*C;zeros(n,1)];
        Aeq = ones(1,n);
        beq = 1;
        %x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)
        solution =quadprog(H,f,A,b,Aeq,beq,[],[],[],options);
        if (isempty(solution)), continue;end
        alpha = solution;
        
        ind = (alpha>epsilon & alpha<C); % Get SV
        if(isempty(ind)),continue;end
        K1 = K(ind,ind');
        rho1 = K1*alpha(ind,1);    
        rho = mean(rho1);                  % Get \rho
        TrainResult = K*alpha-rho;
        TrainCorr = 100*sum(TrainResult>=0)/n;
        if (TrainCorr>=Corr(1))
            Corr(1) = TrainCorr; C0 = C;sigma0 = sigma;
            if (min(TrainResult) < Compactness)
                Compactness = min(TrainResult);
                C0 = C;sigma0 = sigma;
                alpha0 = alpha;rho0 = rho;
            end            
        end 
    end
end

   K_Te = RBFkernel(TeX,X,sigma0,2);
   TestResult = K_Te*alpha0 - rho0;
   Corr(2) = 100*sum(TestResult>=0)/size(TeX,1);
end


function [C0,sigma0,alpha0,rho0,Corr,Compactness] = ParaTuning(TrX,TeX)
% gamma = 1/(2*sigma^2);
n = size(TrX,1); 
X = TrX;
options = optimoptions(@linprog,'Algorithm','interior-point','Display', 'none');
Corr = [0;0]; Compactness = 1e10;
for j = -2:2
        gamma = 10^j; sigma = sqrt(gamma/2);
        %sigma = 15;
    for i = -3:3
        nv = 10^i; C = 1/(nv*n);   
        %C = 10^i;
        K = RBFkernel(X,X,sigma,1);
        f = [ones(2*n,1); C*ones(n,1); -1];
        A = [-K K -eye(n) ones(n,1)];
        A = [A; -eye(n) zeros(n,2*n+1)];
        A = [A;zeros(n,n) -eye(n) zeros(n,n+1)];
        A = [A;zeros(n,2*n) -eye(n) zeros(n,1)];
        b = zeros(4*n,1);
        Aeq = [ones(1,n) -ones(1,n) zeros(1,n+1)];
        beq = 1;
        solution = linprog(f,A,b,Aeq,beq,[],[],options);
        if (isempty(solution)), continue;end
        alpha = solution(1:n,1)-solution(n+1:2*n,1);
        rho = solution(end,1);        
        
        TrainResult = K*alpha-rho;
        TrainCorr = 100*sum(TrainResult>=0)/n;
        if (TrainCorr>=Corr(1))
            Corr(1) = TrainCorr; C0 = C;sigma0 = sigma;
            if (min(TrainResult) < Compactness)
                Compactness = min(TrainResult);
                C0 = C;sigma0 = sigma;
                alpha0 = alpha;rho0 = rho;
            end            
        end 
    end
end

   K_Te = RBFkernel (TeX,X,sigma0,2);
   TestResult = K_Te*alpha0 - rho0;
   Corr(2) = 100*sum(TestResult>=0)/size(TeX,1);
end


function p = show_contourSVDD(X,alpha,ker,kerpar,RR,str,width,shift,a)


[x,y] = meshgrid( a(1)-shift : (a(2)-a(1))/20 : a(2)+shift, a(3)-shift : (a(4)-a(3))/20 : a(4)+shift );
[row,col] = size(x);
newX = [x(:) y(:)];
TestK = RBFkernel(newX,X,kerpar,2);
K = RBFkernel(X,X,kerpar,1);
z = RR-1-alpha'*K*alpha + 2*TestK*alpha;

z = reshape(z,[row col]);
hold on;
p = contour(x,y,z,[0 0],str,'LineWidth',width);%,'DisplayName',displayname);
% contour(x,y,z,[1 1],'g--');
% contour(x,y,z,[-1 -1],'g--');
end

function p = show_contour(X,alpha,ker,kerpar,b,str,width,shift,a)


[x,y] = meshgrid( a(1)-shift : (a(2)-a(1))/20 : a(2)+shift, a(3)-shift : (a(4)-a(3))/20 : a(4)+shift );
[row,col] = size(x);
newX = [x(:) y(:)];
TestK = RBFkernel(newX,X,kerpar,2);
z = TestK*alpha - repmat(b,row*col,1);

z = reshape(z,[row col]);
hold on;
p = contour(x,y,z,[0 0],str,'LineWidth',width);%,'DisplayName',displayname);
% contour(x,y,z,[1 1],'g--');
% contour(x,y,z,[-1 -1],'g--');
end


function K = RBFkernel (X,Y,sigma,flag)
if (flag == 1) % training K
    dis = pdist(X);
    dis = squareform(dis);
    dis = dis.*dis;    
    
else           % testing K  
    dis = pdist2(X,Y);
    dis = dis.*dis;
    
end
K = exp(-dis/(2*sigma*sigma));
end