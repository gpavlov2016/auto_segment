#!/usr/bin/python3
'''
	Copyright 2017 Muhammed Shameem mailtoshameempk@gmail.com

	 This file is part of Fast_seg.

	 Fast_seg is free software: you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published by
	 the Free Software Foundation, either version 3 of the License, or
	 (at your option) any later version.

	 Fast_seg is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 GNU General Public License for more details.

	 You should have received a copy of the GNU General Public License
	 along with Fast_seg.  If not, see <http://www.gnu.org/licenses/>.

'''
import sys, getopt
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import boykov_kolmogorov

import yappi

drawing = False
mode = "ob"
marked_ob_pixels=[]
marked_bg_pixels=[]
I=None
I_dummy=None
l_range=[0,256]
a_range=[0,256]
b_range=[0,256]
lab_bins=[32,32,32]

class SPNode():
	"""docstring for SPNode"""
	def __init__(self):
		self.label=None
		self.pixels=[]
		self.mean_intensity=0.0
		self.centroid=()
		self.type='na'
		self.mean_lab=None
		self.lab_hist=None
		self.real_lab=None
	def __repr__(self):
		return str(self.label)
	
def cie_de_2000(lab1,lab2,k_L,k_C,k_H):
	L_1_star,a_1_star,b_1_star=lab1
	L_2_star,a_2_star,b_2_star=lab2
	C_1_star=np.sqrt(np.power(a_1_star,2)+np.power(b_1_star,2))
	C_2_star=np.sqrt(np.power(a_2_star,2)+np.power(b_2_star,2))
	C_bar_star=np.average([C_1_star,C_2_star])
	
	G=0.5*(1-np.sqrt(np.power(C_bar_star,7)/(np.power(C_bar_star,7)+np.power(25,7))))
	
	a_1_dash=(1+G)*a_1_star
	a_2_dash=(1+G)*a_2_star
	C_1_dash=np.sqrt(np.power(a_1_dash,2)+np.power(b_1_star,2))
	C_2_dash=np.sqrt(np.power(a_2_dash,2)+np.power(b_2_star,2))
	h_1_dash=np.degrees(np.arctan2(b_1_star,a_1_dash))
	h_1_dash += (h_1_dash < 0) * 360
	h_2_dash=np.degrees(np.arctan2(b_2_star,a_2_dash))
	h_2_dash += (h_2_dash < 0) * 360
	
	delta_L_dash=L_2_star-L_1_star
	delta_C_dash=C_2_dash-C_1_dash
	delta_h_dash=0.0
	
	if(C_1_dash*C_2_dash):
		if(np.fabs(h_2_dash-h_1_dash)<=180):
			delta_h_dash=h_2_dash-h_1_dash
		elif(h_2_dash-h_1_dash>180):
			delta_h_dash=(h_2_dash-h_1_dash)-360
		elif(h_2_dash-h_1_dash)<-180:
			delta_h_dash=(h_2_dash-h_1_dash)+360
	
	delta_H_dash=2*np.sqrt(C_1_dash*C_2_dash)*np.sin(np.radians(delta_h_dash)/2.0)
	
	L_bar_dash=np.average([L_1_star,L_2_star])
	C_bar_dash=np.average([C_1_dash,C_2_dash])
	h_bar_dash=h_1_dash+h_2_dash
	
	if(C_1_dash*C_2_dash):
		if(np.fabs(h_1_dash-h_2_dash)<=180):
			h_bar_dash=np.average([h_1_dash,h_2_dash])
		else:
			if(h_1_dash+h_2_dash)<360:
				h_bar_dash=(h_1_dash+h_2_dash+360)/2
			else:
				h_bar_dash=(h_1_dash+h_2_dash-360)/2

	T=1-0.17*np.cos(np.radians(h_bar_dash-30))+0.24*np.cos(np.radians(2*h_bar_dash))\
	+0.32*np.cos(np.radians(3*h_bar_dash+6))-0.20*np.cos(np.radians(4*h_bar_dash-63))
	
	delta_theta=30 * np.exp(- np.power( (h_bar_dash-275) / 25, 2))
	
	R_c=2*np.sqrt( np.power(C_bar_dash,7) / (np.power(C_bar_dash,7)+np.power(25,7)) )
	
	S_L=1+((0.015*np.power(L_bar_dash-50,2))/np.sqrt(20+np.power(L_bar_dash-50,2)))
	S_C=1+0.045*C_bar_dash
	S_H=1+0.015*C_bar_dash*T
	R_T=-R_c * np.sin(2*np.radians(delta_theta))
	
	return np.sqrt(np.power(delta_L_dash/(k_L*S_L),2)+\
		np.power(delta_C_dash/(k_C*S_C),2)+\
		np.power(delta_H_dash/(k_H*S_H),2)+\
		R_T*(delta_C_dash/(k_C*S_C))*(delta_H_dash/(k_H*S_H))\
		)


def mark_seeds(event,x,y,flags,param):
	global drawing,mode,marked_bg_pixels,marked_ob_pixels,I_dummy
	h,w,c=I_dummy.shape

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == "ob":
				if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
					marked_ob_pixels.append((y,x))
				cv2.line(I_dummy,(x-3,y),(x+3,y),(0,0,255))
			else:
				if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
					marked_bg_pixels.append((y,x))
				cv2.line(I_dummy,(x-3,y),(x+3,y),(255,0,0))
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == "ob":
			cv2.line(I_dummy,(x-3,y),(x+3,y),(0,0,255))
		else:
			cv2.line(I_dummy,(x-3,y),(x+3,y),(255,0,0))
	

def gen_sp_seed(I,h,w,c):
	# Superpixel Generation :: Superpixels extracted via energy-driven sampling
	num_sp=500
	num_iter=4
	num_block_levels=1

	sp_seeds=cv2.ximgproc.createSuperpixelSEEDS(w, h, c, num_sp, num_block_levels, prior = 2, histogram_bins=5, double_step = False)
	sp_seeds.iterate(I,num_iterations=num_iter)
	
	return sp_seeds

def gen_sp_slic(I, region_size_):
	# Superpixel Generation ::  Slic superpixels compared to state-of-the-art superpixel methods
	SLIC=100
	SLICO=101
	num_iter=4
	sp_slic=cv2.ximgproc.createSuperpixelSLIC(I, algorithm=SLICO, region_size=region_size_, ruler=10.0)
	sp_slic.iterate(num_iterations=num_iter)

	return sp_slic

def draw_sp_mask(I,SP):
	I_marked=np.zeros(I.shape)
	I_marked=np.copy(I)
	mask=SP.getLabelContourMask()
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j]==-1 or mask[i][j]==255: # SLIC/SLICO marks borders with -1 :: SEED marks borders with 255
				I_marked[i][j]=[128,128,128]
	return I_marked

def draw_centroids(I, SP_list):
	for each in SP_list:
		i,j=each.centroid
		I[i][j]=128
	return I

def draw_edges(I, G):
	for each in G.edges():
		cv2.line(I,each[0].centroid[::-1],each[1].centroid[::-1],128)
	return I

def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def distance_3d(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2]-p1[2])**2)

def dist_matrix(SP_list):
	import scipy
	from scipy.spatial import distance_matrix
	M = [v.centroid for v in SP_list]
	N = [v.centroid for v in SP_list]
	res = scipy.spatial.distance_matrix(M, N)
	return res

def comp(u,v):
	return cv2.compareHist(u.lab_hist, v.lab_hist, 3)

def calc(comph, sig_, dist):
	return math.exp(-(comph**2/2*sig_**2))*(1/dist)

def add_edges(G, SP_list, D, i):
	u = SP_list[i]
	K = 0
	sig_=5
	region_rad = math.sqrt(len(u.pixels) / math.pi)
	for j in range(len(SP_list)):
		v = SP_list[j]
		if u != v:
			dist = D[i][j]  # distance(u.centroid, v.centroid)
			if dist <= 2.5 * region_rad:
				sim = G.get_edge_data(v, u)
				if sim != None:
					sim = sim['sim']
				else:
					comph = comp(u, v)
					sim = calc(comph, sig_, dist)
					# sim=math.exp(-(comph**2/2*sig_**2))*(1/dist)
				K += sim
				G.add_edge(u, v, sim=sim)
	return K



def gen_graph(I,SP_list,hist_ob,hist_bg):
	G=nx.Graph()
	s=SPNode()
	s.label='s'
	t=SPNode()
	t.label='t'
	lambda_=.9
	hist_ob_sum=int(hist_ob.sum())
	hist_bg_sum=int(hist_bg.sum())

	D = dist_matrix(SP_list)

	sp1 = sorted(SP_list, key=lambda x: x.centroid[0])
	sp2 = sorted(SP_list, key=lambda x: x.centroid[1])
	print(sp1)
	print(sp2)
	for i in range(len(SP_list)):
		u = SP_list[i]
		K = add_edges(G, SP_list, D, i)
		if(u.type=='na'):
			l_,a_,b_=[int(x) for x in u.mean_lab]
			l_i=int(l_//((l_range[1]-l_range[0])/lab_bins[0]))
			a_i=int(a_//((a_range[1]-a_range[0])/lab_bins[1]))
			b_i=int(b_//((b_range[1]-b_range[0])/lab_bins[2]))
			pr_ob=int(hist_ob[l_i,a_i,b_i])/hist_ob_sum
			pr_bg=int(hist_bg[l_i,a_i,b_i])/hist_bg_sum
			sim_s=100000
			sim_t=100000
			if pr_bg > 0:
				sim_s=lambda_*-np.log(pr_bg)
			if pr_ob > 0:
				sim_t=lambda_*-np.log(pr_ob)
			G.add_edge(s, u, sim=sim_s)
			G.add_edge(t, u, sim=sim_t)
		if(u.type=='ob'):
			G.add_edge(s, u, sim=1+K)
			G.add_edge(t, u, sim=0)
		if(u.type=='bg'):
			G.add_edge(s, u, sim=0)
			G.add_edge(t, u, sim=1+K)		
	return G

def main():
	global I,mode,I_dummy

	inputfile = ''
	try:
		opts, args = getopt.getopt(sys.argv[1:], "i:h", ["input-image=", "help"])
	except getopt.GetoptError:
		print('fast_seg.py -i <input image>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('fast_seg.py -i <input image>')
			sys.exit()
		elif opt in ("-i", "--input-image"):
			inputfile = arg
	print('Using image: ', inputfile)

	inputfile = 'yellowcubes.jpg'

	I=cv2.imread(inputfile) #imread wont rise exceptions by default
	I = cv2.resize(I, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

	I_dummy=np.zeros(I.shape)
	I_dummy=np.copy(I)
	
	h,w,c=I.shape
	region_size=20

	# cv2.namedWindow('Mark the object and background')
	# cv2.setMouseCallback('Mark the object and background',mark_seeds)
	# while(1):
	# 	cv2.imshow('Mark the object and background',I_dummy)
	# 	k = cv2.waitKey(1) & 0xFF
	# 	if k == ord('o'):
	# 		mode = "ob"
	# 	elif k == ord('b'):
	# 		mode = "bg"
	# 	elif k == 27:
	# 		break
	# cv2.destroyAllWindows()

	import time

	s = time.clock()

	I_lab=cv2.cvtColor(I, cv2.COLOR_BGR2Lab)
	SP=gen_sp_slic(I,region_size)
	SP_labels=SP.getLabels()
	SP_list=[None for each in range(SP.getNumberOfSuperpixels())]

	s1 = time.clock()
	print('time 1 - ', s1-s)

	for i in range(h):
		for j in range(w):
			if not SP_list[SP_labels[i][j]]:
				tmp_sp=SPNode()
				tmp_sp.label=SP_labels[i][j]
				tmp_sp.pixels.append((i,j))
				SP_list[SP_labels[i][j]]=tmp_sp
			else:
				SP_list[SP_labels[i][j]].pixels.append((i,j))

	s2 = time.clock()
	print('time 2 - ', s2 - s1)

	for sp in SP_list:
		n_pixels=len(sp.pixels)
		i_sum=0
		j_sum=0
		lab_sum=[0,0,0]
		tmp_mask=np.zeros((h,w),np.uint8)
		for each in sp.pixels:
			i,j=each
			i_sum+=i
			j_sum+=j
			#lab_sum=[x + y for x, y in zip(lab_sum, I_lab[i][j])]
			lab_sum +=  I_lab[i][j]
			tmp_mask[i][j]=255
		# print(lab_sum)
		sp.lab_hist=cv2.calcHist([I_lab],[0,1,2],tmp_mask,lab_bins,l_range+a_range+b_range)
		sp.centroid+=(i_sum//n_pixels, j_sum//n_pixels,)
		sp.mean_lab=[x/n_pixels for x in lab_sum]
		sp.real_lab=[sp.mean_lab[0]*100/255,sp.mean_lab[1]-128,sp.mean_lab[2]-128]

	s3 = time.clock()
	print('time 3 - ', s3 - s2)

	# print(marked_ob_pixels)
	# print(marked_bg_pixels)
	marked_ob_pixels = [(323, 389), (323, 389)]#[[h//22, w//2], [h//22+5, w//2+5]]
	marked_bg_pixels = [(84, 330), (84, 331), (84, 332), (84, 333), (84, 334), (84, 335), (84, 336), (84, 337)]#[[1, 1], [h-1, w-1], [1, w-1], [h-1, 1]]
	for pixels in marked_ob_pixels:
		x,y = pixels
		SP_list[SP_labels[x][y]].type="ob"
	for pixels in marked_bg_pixels:
		x,y = pixels
		SP_list[SP_labels[x][y]].type="bg"
	I_marked=np.zeros_like(I)
	#I_marked =draw_sp_mask(I,SP)
	I_marked=draw_centroids(I_marked,SP_list)
	mask_ob=np.zeros((h,w),dtype=np.uint8)
	for pixels in marked_ob_pixels:
		i,j=pixels
		mask_ob[i][j]=255
	mask_bg=np.zeros((h,w),dtype=np.uint8)
	for pixels in marked_bg_pixels:
		i,j=pixels
		mask_bg[i][j]=255

	s4 = time.clock()
	print('time 4 - ', s4 - s3)

	hist_ob=cv2.calcHist([I_lab],[0,1,2],mask_ob,lab_bins,l_range+a_range+b_range)

	hist_bg=cv2.calcHist([I_lab],[0,1,2],mask_bg,lab_bins,l_range+a_range+b_range)
	G=gen_graph(I_lab,SP_list,hist_ob,hist_bg)

	s5 = time.clock()
	print('time 5 - ', s5 - s4)

	for each in G.nodes():
		if each.label=='s':
			s=each
		if each.label=='t':
			t=each

	RG=boykov_kolmogorov.boykov_kolmogorov(G, s, t, capacity='sim')
	source_tree, target_tree = RG.graph['trees']
	partition = (set(source_tree), set(G) - set(source_tree))
	F=np.zeros((h,w),dtype=np.uint8)
	for sp in partition[0]:
		for pixels in sp.pixels:
			i,j=pixels
			F[i][j]=1
	Final=cv2.bitwise_and(I,I,mask = F)

	s6 = time.clock()
	print('time 6 - ', s6 - s5)


	sp_lab=np.zeros(I.shape,dtype=np.uint8)
	for sp in SP_list:
		for pixels in sp.pixels:
			i,j=pixels
			sp_lab[i][j]=sp.mean_lab
	sp_lab=cv2.cvtColor(sp_lab, cv2.COLOR_Lab2RGB)
	
	plt.subplot(2,2,1)
	plt.tick_params(labelcolor='black', top='off', bottom='off', left='off', right='off')
	plt.imshow(I[...,::-1])
	plt.axis("off")
	plt.xlabel("Input image")

	plt.subplot(2,2,2)
	plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	plt.imshow(I_marked[...,::-1])
	plt.axis("off")
	plt.xlabel("Super-pixel boundaries and centroid")

	plt.subplot(2,2,3)
	plt.imshow(sp_lab)
	plt.axis("off")
	plt.xlabel("Super-pixel representation")


	plt.subplot(2,2,4)
	plt.imshow(Final[...,::-1])
	plt.axis("off")
	plt.xlabel("Output Image")


	
	cv2.imwrite("out.png",Final)
	plt.show()

if __name__ == '__main__':
	yappi.start()
	main()
	yappi.get_func_stats().print_all()
	yappi.get_thread_stats().print_all()
