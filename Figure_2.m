    clc;clear;close all

    load('data.mat')
    nT2=128;
    T2min=1.0e-2;     T2max=1.0e4;
    T22=logspace(log10(T2min),log10(T2max),nT2);

    figure
    semilogx(T22,T2_dist_s,'r-','linewidth',3.5);hold on;
    semilogx(T22,T2_dist_c,'g-','linewidth',3.5);hold on;
    xlabel('{\itT}_2 (ms)','FontName','Arial','fontsize',20,'fontweight','bold')
    ylabel('Porosity (%)','FontName','Arial','fontsize',20,'fontweight','bold')
    set(gca,'fontsize',25,'fontweight','bold','fontname','Times New Roman')
    xlim([T2min T2max])
    set(gca,'XTick',[0.01 0.1 1 10 100 1000 10000])

    %%
%      Please note that the setting of 6 rows here is only for the purpose of visualization; 
%      The actual form of Type 1 data consists of 5 rows.
    data_type1=zeros(6,nT2);  
    data_type1(2,:)=T2_dist_s;
    data_type1(4,:)=T2_dist_c;
    figure
    surf(data_type1);
    view([0 90]);
    xlim([1 128]);        ylim([1 6]);
    colormap jet;
    set(gca, 'XTick', []);        set(gca, 'YTick', []);

    %%
    data_type2=zeros(1,nT2*2);
    data_type2(1:nT2)=T2_dist_s;
    data_type2(nT2+1:2*nT2)=T2_dist_c;

    figure
    plot((1:nT2),  data_type2(1:nT2),'r-','linewidth',3.5);hold on;
    plot((nT2+1:2*nT2),data_type2(nT2+1:2*nT2),'g-','linewidth',3.5);hold on;
    axis off
    set(gca, 'XTick', []);        set(gca, 'YTick', []);
    xlim([1 2*nT2])

    %%
    max_max=0.7639;
    dpi=30;
    figure
    semilogx(T22,T2_dist_s,'k-',T22,T2_dist_c,'k-','linewidth',5.5)
    ylim([0 max_max])
    set(gca, 'box', 'off', 'XTick', [], 'YTick', []);
    axis off;
    filename = '/Fig_type_3.jpg';
    print(gcf, fullfile(cd, filename), '-djpeg', ['-r', num2str(dpi)]);











