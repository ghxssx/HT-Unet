#!/bin/sh
source ~/.bashrc

ncl << EOF
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;************************************************
begin

dir_wrf_ctrl="/public/gh/result/SO2_replace/2023111918"
dir_picture="/public/gh/result/SO2_replace/"

  type = "png"
  xx=100
  yy=100
  finames = systemfunc("cd "+dir_wrf_ctrl+"/;ls cha_wrfrst_d01_2023-11-19_18:00:00") 
  nfil = dimsizes( finames )

  ; 定义等分的色带范围
  arr=(/-5,-2,-1,-0.5,-0.2,0,0.2,0.5,1,2,5/)
  SO2=new((/xx,yy/),double,-999)
do nf=0,0;0,nfil-1
  a = addfile(dir_wrf_ctrl+"/"+finames(nf),"r")
  times = wrf_user_getvar(a,"times",-1)   
  print("Time is = "+times)

  lat=a->XLAT(0,:,:)
  lon=a->XLONG(0,:,:)
  so2_ctrl=a->SO2(0,0,:,:)
  ; dz8w=a->DZ8W(0,0:59,:,:)
print(dimsizes(so2_ctrl))
so2_ctrl = so2_ctrl*64064/22.414
print(max(so2_ctrl))
print(min(so2_ctrl))

  wks = gsn_open_wks(type,dir_picture+"cha_50_SO2_71"+times)
  ;gsn_define_colormap(wks,"WhViBlGrYeOrRe")
  ;gsn_define_colormap(wks,"cmp_b2r") 
  gsn_define_colormap(wks,"BlueRed")
  ;======================China  map=========================
  mpres                             = True
  mpres@gsnDraw                     = False              ; don't draw
  mpres@gsnFrame                    = False              ; don't advance frame
  mpres@gsnMaximize                 = False 
  mpres@mpGridLineColor             = "Black"
  mpres@mpLimbLineColor             = "Black"
  mpres@mpPerimLineColor            = "Black"
  mpres@mpUSStateLineColor          = "Black"
  mpres@pmTickMarkDisplayMode       = "Always"         ; turn on tickmarks
  mpres@tmXTOn                      = False
  mpres@tmXBMinorOn                 = False
  mpres@tmXTMinorOn                 = False
  mpres@tmYLMinorOn                 = False
  mpres@tmYRMinorOn                 = False
  mpres@mpDataSetName               = "Earth..4"
  mpres@mpDataBaseVersion           = "MediumRes" ; or "Ncarg4_1"
  mpres@mpFillBoundarySets          = "AllBoundaries"
  mpres@mpOutlineBoundarySets       = "AllBoundaries"
  mpres@mpNationalLineColor         = "black"
  mpres@mpProvincialLineColor       = "black"
  mpres@mpGeophysicalLineColor      = "black"
  mpres@mpNationalLineThicknessF    = 3
  mpres@mpProvincialLineThicknessF  = 3
  mpres@mpGeophysicalLineThicknessF = 3
  mpres@tmXBLabelFontHeightF   = 0.018
  mpres@tmYLLabelFontHeightF   = 0.018
  mpres@tmXBMajorLengthF       = -0.021
  mpres@tmXBMinorLengthF       = -0.008
  mpres@tmYLMajorLengthF       = -0.021
  mpres@tmYLMinorLengthF       = -0.008
  mpres@tmXBMajorOutwardLengthF = -0.021
  mpres@tmXBMinorOutwardLengthF = -0.008
  mpres@tmYLMajorOutwardLengthF = -0.021
  mpres@tmYLMinorOutwardLengthF = -0.008


plotmap=wrf_map(wks,a,mpres)
plotmap_1=wrf_map(wks,a,mpres)


  res=True
  res@gsnDraw                       = False              ; don't draw
  res@gsnFrame                      = False              ; don't advance frame
  res@gsnMaximize                 = False
  res@tfDoNDCOverlay                = True
  res@tmXTOn = False            ; turn off top   labels
  res@tmYROn = False            ; turn off right labels
  res@gsnRightString                = ""
  res@gsnLeftString                 = ""
    ; Plotting options for  PM2.5 
      cnres = res
      ;cnres@gsnDraw               = False              ; don't draw
      ;cnres@gsnFrame              = False              ; don't advance frame
      cnres@cnFillOn              = True             ; color plot desired
      cnres@cnLinesOn             = False            ; turn off contour lines
      cnres@cnLineLabelsOn        = False            ; turn off contour labels
      cnres@cnFillDrawOrder       = "PreDraw"
      cnres@cnInfoLabelOn         = False
      ; cnres@gsnLeftString         = "SO~B~2~N~~B~"
      ;cnres@gsnRightString        = "DU";
      cnres@cnLevelSelectionMode  = "ExplicitLevels"
      cnres@cnLevels              = arr  
      ;cnres@cnFillMode           = "RasterFill"
      ;res@lbLabelBarOn           =False 
      ;cnres@pmLabelBarOrthogonalPosF = -0.03  ;垂直
      ;cnres@pmLabelBarHeightF      = 0.14   ;高度
      ;cnres@lbLabelFontHeightF     = 0.016  ;字体
      ;cnres@cnFillPalette          = "grads_rainbow"
  
    plotpm_con = gsn_csm_contour(wks,so2_ctrl,cnres)
    overlay(plotmap, plotpm_con)
	
  draw(plotmap)
  frame(wks) 
end do
end
