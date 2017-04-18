from PIL import Image
import numpy
import os
import scipy.ndimage
from scipy.linalg import lstsq
import math
import random
import time
from pylab import plt as plt
from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
import matplotlib
import csv
from check_phase import phase_agreement
import scipy.misc
import matplotlib as mpl


sigma=8
funct_type=0
# funct_type=0 dla kazdego emitera stawia kwadrat -sigma,+sigma, czyli o boku 2*sigma+1
# funct_type=1 dla kazdego emitera stawia gaussa o szerokosci -sigma,+sigma

max_emitters_nr=50000000
min_emitters_nr=50000

#ilosc emiterow

input_file=("test2.tif")
#plik wejsciowy, wyznacza gestosc emiterow

output_dir_name="results_logtest2_sigma8_50M"

showlog=True

accuracy=25e-9
#dokladnosc wyznaczeniq wspolrzednych emitera

frame_nr=100

basic_resolution=6.4e-9
#wymiar piksela

dump_plots=True

runs_nr=1

class Simulate:
    def __init__(self,runs_nr=1):
        self.subdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),output_dir_name)
        self.corr=[]
        self.runs=runs_nr
        self.binary=[]
        self.l2_norms=[]
        self.ft_data=[]
        self.accuracy=accuracy/basic_resolution
        scale_step=(numpy.log(max_emitters_nr)-numpy.log(min_emitters_nr))/(frame_nr-1)
        self.frame_to_emm_nr=map(lambda x:int(round(x,0)),numpy.exp(numpy.add(numpy.log(min_emitters_nr),numpy.multiply(scale_step,range(frame_nr)))))
        if funct_type==1:
            self.funct=self.gauss
            self.em_range=range(-3*sigma,3*sigma+1)
            self.sigma_2=1.0*sigma**2
        if funct_type==0:
            self.em_range=range(-sigma,sigma+1)
            self.funct=self.square
        self.cmapr = matplotlib.colors.ListedColormap(['black', 'black', 'red', 'red'])
        self.cmapb = matplotlib.colors.ListedColormap(['black', 'black', 'blue', 'blue'])
        if os.path.exists(self.subdir):
            if dump_plots:
                filesToRemove = [f for f in os.listdir(self.subdir)]
                for f in filesToRemove:
                    os.remove(os.path.join(self.subdir,f)) 
        else:
            os.mkdir(self.subdir)
                
                
    class emitter:
        def __init__(self,x,y,accuracy):
            self.accuracy=accuracy
            self.x=x+random.gauss(0,self.accuracy)
            self.y=y+random.gauss(0,self.accuracy)
            self.xr=int(round(self.x,0))
            self.yr=int(round(self.y,0))
    def gauss(self,dx,dy):
        return(numpy.exp(-(dx**2+dy**2)/self.sigma_2))
    def square(self,dx,dy):
        return(1)
    @staticmethod
    def open_image(f_name):
        im = Image.open(f_name)
        imdata=numpy.array(im.getdata(),dtype=numpy.uint16)
        imdata=imdata.reshape(im.size[1],im.size[0])
        imdata=numpy.multiply(1.0/numpy.max(imdata),imdata)
        return(imdata)
    def format_labels(self,ax):
        def get_text(x):
            return(float(x._text))
        print map(get_text,ax.get_xticklabels())
    def mjrFormatter(self,x,pos):
                return str(basic_resolution*int(x))
    def mjrFormattera(self,x,pos):
                number=1e-12*int(x)/self.area
                if ((number>1000)|(number<0.001)):
                    return "{0:.2e}".format(number)
                else:
                    return "{0:.2f}".format(number)
    def mjrFormatterb(self,x,pos):
                if x<0:
                    x=0
                print "index:",x
                try:
                    number=1e-15*self.frame_to_emm_nr[int(x)]/(self.area)
                except(IndexError):
                    number=1e-15*self.frame_to_emm_nr[-1]/(self.area)
                if ((number>1000)|(number<0.001)):
                    nr="{0:.1e}".format(number)
                else:
                    nr="{0:.1f}".format(number)
                print "nr=",nr
                return nr
    def mjrFormatterc(self,x,pos):
                number=self.ftfactor*int(x)
                if ((number>1000)|(number<0.001)):
                    return "{0:.2e}".format(number)
                else:
                    return "{0:.2f}".format(number)
    
    def calculate(self):
        s_corr=[]
        l2_norms=[]
        s_binary=[]
        self.emitters_set=[]
        self.em_nr=[]
        imdata = Simulate.open_image(input_file)
        imshape=imdata.shape
        print imshape
        self.xsize=basic_resolution*imshape[0]
        self.ysize=basic_resolution*imshape[1]
        self.area=self.xsize*self.ysize
        self.maxfreq=0.5*math.sqrt(2)/basic_resolution
        im_out=numpy.zeros([imshape[0],imshape[1]])
        xmax=imshape[0]-1
        ymax=imshape[1]-1
        while len(self.emitters_set)<max_emitters_nr:
            x_em=random.uniform(0,xmax)
            y_em=random.uniform(0,ymax)
            x_coord=int(round(x_em,0))
            y_coord=int(round(y_em,0))
            prob=random.uniform(0,1)
            if imdata[x_coord,y_coord]>prob:
                self.emitters_set.append(self.emitter(x_em,y_em,self.accuracy))
            
                lem=len(self.emitters_set)
                if (lem%10000==0):
                    print "Emitters added:",lem
         
        pt=max_emitters_nr/frame_nr

        for current_frame in range(frame_nr):
            maxem=self.frame_to_emm_nr[current_frame]
            im_out=numpy.zeros([imshape[0],imshape[1]])
            print maxem
            for em in self.emitters_set[:maxem]:
                for dx in self.em_range:
                    for dy in self.em_range:
                        xi=em.xr+dx
                        yi=em.yr+dy
                        
                        try:
                            #print dx,dy,gauss(dx,dy)
                            im_out[xi,yi]=im_out[xi,yi]+self.funct(dx,dy)
                        except(IndexError):
                            pass
                            
       
            coeff=numpy.corrcoef(imdata.flatten(),im_out.flatten())[0][1]
            print "Correlation coeff.:",coeff
            
            threshold_global_otsu = threshold_otsu(imdata)
            global_otsu = numpy.array(imdata >= threshold_global_otsu,dtype=numpy.int)
            threshold_global_otsu_out = threshold_otsu(im_out)
            global_otsu_out = numpy.array(im_out >= threshold_global_otsu_out,dtype=numpy.int)
            diff=numpy.abs(numpy.subtract(global_otsu,global_otsu_out))
            binary=1.0-numpy.mean(diff)
            print "Binary agreement:",binary
            imdata_std=numpy.std(imdata)
            imout_std=numpy.std(im_out)
            imdata_normed=numpy.multiply(1.0/imdata_std,numpy.subtract(imdata,-numpy.mean(imdata)))
            imout_normed=numpy.multiply(1.0/imout_std,numpy.subtract(im_out,-numpy.mean(im_out)))
            l2norm=numpy.sqrt(numpy.mean(numpy.power(numpy.subtract(imdata_normed,imout_normed),2)))
            print "L2norm:",l2norm
            if runs_nr==1:
                self.ft_data.append(numpy.abs(phase_agreement(imdata_normed,imout_normed)))
            l2_norms.append(l2norm)
            s_corr.append(coeff)
            s_binary.append(binary)
            self.em_nr.append(maxem)
            
            if dump_plots:
                fig2 = plt.figure(figsize=[20,20])
                ax2 = fig2.add_subplot(111)
                ax2.imshow(im_out)
                ax2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
                ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
                fig2.savefig(os.path.join(self.subdir,"figure"+str(maxem)+".tif"))
                scipy.misc.imsave(os.path.join(self.subdir,"output"+str(maxem)+".tif"), im_out)
                plt.close(fig2)
                
                fig5 = plt.figure(figsize=[20,20])
                ax5 = fig5.add_subplot(111)
                ax5.imshow(global_otsu_out,cmap=self.cmapr,alpha=0.5)
                ax5.imshow(global_otsu,cmap=self.cmapb,alpha=0.7)
                ax5.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
                ax5.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
                fig5.savefig(os.path.join(self.subdir,"binary"+str(maxem)+".tif"))
                plt.close(fig5)
        self.corr.append(s_corr)
        self.binary.append(s_binary)
        self.l2_norms.append(l2_norms)
        self.imdata=imdata
        self.im_out=im_out
        self.diff=[global_otsu,global_otsu_out]
    def show_plots(self):    
        fig1 = plt.figure(figsize=[20,20])
        ax1 = fig1.add_subplot(111)
        ax1.imshow(self.imdata)
        ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
        ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
        fig1.savefig(os.path.join(self.subdir,"figure"+str(9999999999999999999999)+".tif"))
        fig2 = plt.figure(figsize=[10,10])
        ax2 = fig2.add_subplot(111)
        ax2.imshow(self.im_out)
        ax2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
        ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
        
        fig3 = plt.figure(figsize=[10,10])
        ax3 = fig3.add_subplot(111)
        ax3.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormattera))
        ax3.set_xlabel("Emitters density $(1/ \mu m ^2)$")
        if showlog:
            ax3.set_xscale('log')
        ax3.set_ylim([0.0,1.0])
        ax3a = ax3.twinx()
        if self.runs==1:
            ax3.plot(self.em_nr,self.corr[0],ls="-",color="green",marker="None",label="correlation")
            ax3.plot(self.em_nr,self.binary[0],ls="-",color="red",marker="None",label="binary agreement")
            ax3a.plot(self.em_nr,self.l2_norms[0],ls="-",color="blue",marker="None",label="L2 norm")
            self.data=numpy.array([numpy.multiply(1.0e-12/self.area,self.em_nr),self.corr[0],self.binary[0],self.l2_norms[0]]).transpose()
        else:
            ax3.plot(self.em_nr,numpy.mean(self.corr,axis=0),ls="-",color="green",marker="None",label="correlation")
            ax3.plot(self.em_nr,numpy.mean(self.binary,axis=0),ls="-",color="red",marker="None",label="binary agreement")
            ax3a.plot(self.em_nr,numpy.mean(self.l2_norms,axis=0),ls="-",color="blue",marker="None",label="L2 norm")
            ax3.errorbar(self.em_nr,numpy.mean(self.corr,axis=0),numpy.std(self.corr,axis=0),color="green")
            ax3.errorbar(self.em_nr,numpy.mean(self.binary,axis=0),numpy.std(self.binary,axis=0),color="red")
            ax3a.errorbar(self.em_nr,numpy.mean(self.l2_norms,axis=0),numpy.std(self.binary,axis=0),color="blue")
            self.data=numpy.array([numpy.multiply(1.0e-12/self.area,self.em_nr),numpy.mean(self.corr,axis=0),numpy.std(self.corr,axis=0),numpy.mean(self.binary,axis=0),numpy.std(self.binary,axis=0),numpy.mean(self.l2_norms,axis=0),numpy.std(self.l2_norms,axis=0)]).transpose()
        ax3.grid("on")
        ax3.legend(loc="upper left")
        ax3a.legend()
        fig5 = plt.figure(figsize=[10,10])
        ax5 = fig5.add_subplot(111)
        ax5.imshow(self.diff[1],cmap=self.cmapr,alpha=0.8)
        ax5.imshow(self.diff[0],cmap=self.cmapb,alpha=0.6)
        ax5.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
        ax5.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatter))
        fig3.savefig(os.path.join(self.subdir,"qmeasures.tif"))
        
        if runs_nr==1:
            fig6 = plt.figure(figsize=[12,14])
            ax6 = fig6.add_subplot(111)
            self.ft_data=numpy.array(self.ft_data)
            self.ftfactor=self.maxfreq/(self.ft_data.shape[1])
            cax6=ax6.imshow(self.ft_data.transpose(),vmin=0.0,vmax=1.0)
            ax6.set_xticks(numpy.arange(-0.5, self.ft_data.shape[0]+0.5*self.ft_data.shape[0]/10, self.ft_data.shape[0]/10));
            ax6.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatterb))
            ax6.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.mjrFormatterc))
            ax6.set_xlabel("Emitters density $[1000/ \mu m ^2]$")
            ax6.set_ylabel("Spatial frequency $[1/m]$")
            ax6ylims=ax6.get_ylim()
            #print ax6.get_xlim()
            ax6.set_ylim([ax6ylims[0]-0.65*(ax6ylims[0]-ax6ylims[1]),ax6ylims[1]])
            fig6.autofmt_xdate(rotation=90)
            fig6.savefig(os.path.join(self.subdir,"transfunc.tif"))
            plt.colorbar(cax6)
            transfunc_output=os.path.join(self.subdir,"transfunc.txt")
            numpy.savetxt(transfunc_output,self.ft_data.transpose(),delimiter=",")
        
        #fig4 = plt.figure(figsize=[10,10])
        #ax4a = fig4.add_subplot(211)
        #ax4b = fig4.add_subplot(212)
        #ax4a.hist(self.imdata.flatten(),bins=10)
        #ax4b.hist(self.im_out.flatten(),bins=10)
        
        #ax5a.imshow(diff,cmap='jet')
        self.save_results()
        plt.show()
    def save_results(self):
        f=open(os.path.join(self.subdir,"results.csv"),"wb")
        f_csv=csv.writer(f)
        for line in self.data:
            f_csv.writerow(line)
        f.close()
            
        

e=Simulate(runs_nr=runs_nr)
for i in range(e.runs):
    e.calculate()
e.show_plots()

