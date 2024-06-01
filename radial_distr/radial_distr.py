"""Provide the primary functions."""
import argparse
import numpy as np
import parmed as pmd
import pandas as pd
import gridData
import numba

def main():
    args = get_args()    
    #read_mrcfile(), absolute path was needed, and ala test case had .save filename changed to .dx as .save was not recognized
    #g = gridData.Grid("/home/tyork/ribosome/radial_distr/radial_distr/tests/guv.O.5.dx")
    g = gridData.Grid(args.guv)
    print("grid shape: ", g.grid.shape)
    print("number of dimensions: ", g.grid.ndim)
    print("origin: ", g.origin)

    
    #read_molecule(), parmed either loads both the parm7 and rst7 or just the pdb file per line
    #parm = pmd.load_file("/home/tyork/ribosome/radial_distr/radial_distr/tests/ala.parm7","/home/tyork/ribosome/radial_distr/radial_distr/tests/ala.rst7",)
    parm = pmd.load_file(args.p,args.y)
    print("test print", parm.coordinates)
    print(parm.coordinates.shape)
    rdf,dr = compute_rad_dist_numba(parm.coordinates, g.grid,g.midpoints)
    df = pd.DataFrame({'sep':np.linspace(dr / 2,len(rdf) * dr - dr / 2, len(rdf)), 'rdf':rdf})
    df = df.fillna(0)
    #rdf = compute_rad_dist(parm.coordinates, g)
    print(df)
    df.to_csv('rdf.csv', index=False)

def get_args():
    parser = argparse.ArgumentParser(prog='radial_distr.py',description='This program creates a radial distribution function around a biological molecule')
    parser.add_argument('--guv',help='guv file expected',required=True)
    parser.add_argument('-p',help='parmtop file',required=True)
    parser.add_argument('-y',help='coordinate file',required=True)
    args = parser.parse_args()
    return args
@numba.jit(cache=True,nopython=True, nogil=True,parallel=False,fastmath=True)
def compute_rad_dist_numba(coordinates, grid, midpoints):
    nbins = 200
    dr = 0.1
    
    #create arrays for hist and rdf
    hist = np.zeros((nbins), dtype=np.int64)
    rdf = np.zeros(nbins)
    print("test hist and rdf arrays:", hist, rdf)
    
    for ix in numba.prange(grid.shape[0]):
        for iy in range(grid.shape[1]):
            print(iy)
            for iz in range(grid.shape[2]):
                
                mdpnt = np.array([midpoints[0][ix], midpoints[1][iy], midpoints[2][iz]])
                
                #print("test xyz:", iz, iy, ix, grid[ix,iy,iz]) #will print g(r) value at the given point
                mindist = 999999.0
                for iatom in range(coordinates.shape[0]):
                    #print("Our coordinates are:", iatom, coordinates[iatom],mdpnt,)
                    
                    # match each g(r) value on each gridpoint to each atom and calculate the distance in numpy
                    dist = np.linalg.norm(mdpnt - coordinates[iatom])
                    #print("Our distance is:", dist)
                    #test if this is the min dist
                    
                    if dist < mindist:
                        mindist = dist
                        #print("the minimum distance is", mindist)
                        #find the bin number
                binnum = int(mindist / dr)     
                #print("The bin number is:", binnum)
                if nbins > binnum:
                    rdf[binnum] += grid[ix,iy,iz] #increment by the g(r) values
                    hist [binnum] += 1 
            if iy == 2: return
    #normalize
    rdf = rdf / hist
    #df is data frame
    return rdf,dr
    
def compute_rad_dist(coordinates, grid):
    nbins = 200
    dr = 0.1
    
    #create arrays for hist and rdf
    hist = np.zeros((nbins), dtype=int)
    rdf = np.zeros(nbins)
    #print("test hist and rdf arrays:", hist, rdf)
    
    for ix in range(grid.grid.shape[0]):
        for iy in range(grid.grid.shape[1]):
            for iz in range(grid.grid.shape[2]):
                mdpnt = np.array([grid.midpoints[0][ix], grid.midpoints[1][iy], grid.midpoints[2][iz]])
                
                #print("test xyz:", iz, iy, ix, grid.grid[ix,iy,iz]) #will print g(r) value at the given point
                mindist = 999999.0
                for iatom in range(coordinates.shape[0]):
                    #print("Our coordinates are:", iatom, coordinates[iatom],mdpnt,)
                    
                    # match each g(r) value on each gridpoint to each atom and calculate the distance in numpy
                    dist = np.linalg.norm(mdpnt - coordinates[iatom])
                    #print("Our distance is:", dist)
                    #test if this is the min dist
                    
                    if dist < mindist:
                        mindist = dist
                        #print("the minimum distance is", mindist)
                        #find the bin number
                binnum = int(mindist / dr)     
                #print("The bin number is:", binnum)
                if nbins > binnum:
                    rdf[binnum] += grid.grid[ix,iy,iz] #increment by the g(r) values
                    hist [binnum] += 1 
        break
    #normalize
    rdf = rdf / hist
    #df is data frame
    df = pd.DataFrame({'sep':np.linspace(dr / 2,nbins * dr - dr / 2, nbins), 'rdf':rdf})
    df = df.fillna(0)
    

                

    return df 
#def write_radial_dist():                
            
    
    

# executes as a python script
if __name__ == "__main__":
    main()
    
