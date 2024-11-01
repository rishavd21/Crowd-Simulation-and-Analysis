import math
import cv2
import numpy as np
import cvxpy as cp
class Object:
    def __init__(self, radius, current_position, final_position, optimal_speed, current_velocity, isstill =0):
        self.r = radius
        self.curr_pos = current_position
        self.fin_pos = final_position
        self.opt_speed = optimal_speed
        self.curr_vel = current_velocity
        self.isstill = isstill

tau =7
def dis(obj1r, obj2r):
    position1 = obj1r.curr_pos
    position2 = obj2r.curr_pos
    
    distance = float(math.sqrt((position2[0] - position1[0])**2 + (position2[1] - position1[1])**2))
    # print(position1[1])
    return distance
def mag(pos1, pos2):
    
    distance = float(math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2))
    # print(position1[1])
    return distance
def find_intercepts(slopes, points):
    intercepts = []
    for slope, point in zip(slopes, points):
        x1, y1 = point
        
        # Calculate the y-intercept using the point-slope form of the equation
        intercept = y1 - slope * x1
        intercepts.append(intercept)
    
    return intercepts






def find_closest_point(slopes, intercepts, points, X):
    n = len(slopes)
    x = cp.Variable()
    y = cp.Variable()
    
    constraints = []
    for i in range(n):
        m = slopes[i]
        c = intercepts[i]
        px, py = points[i]
        
        if py > m * px + c:
            constraints.append(y >= m * x + c)
        else:
            constraints.append(y <= m * x + c)
    
    objective = cp.Minimize(cp.square(x - X[0]) + cp.square(y - X[1]))
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except cp.error.SolverError:
        prob.solve(solver=cp.ECOS, verbose=False)
    
    if prob.status == 'optimal':
        closest_point = [x.value, y.value]
    else:
        closest_point = [0,0]
    
    return closest_point





def moveonestep(objects):
    u = (0.0,0.0)
    col= 0
    
    for qw in range(1):
        vel =[]
        for o1 in range(len(objects)):

            obj1 = objects[o1]
            ra = obj1.r
            pa = obj1.curr_pos
            slope= []
            ponline = []
            intercepts = []
            pinreg = []
            fa = obj1.fin_pos
            opt1= (fa[0] - pa[0],fa[1]-pa[1])
            if(obj1.isstill > 1e-1 or obj1.opt_speed< 1e-1 or mag(pa,fa) < 1e-4):
                
                radius1 = 10
                obj1.curr_vel = (0,0)
                vel.append(obj1.curr_vel)

                continue
                
            div = float(obj1.opt_speed/mag(pa,fa))
            opt = tuple(pt1*float(div) for pt1 in opt1)
            u = (0,0)

            for o2 in range(len(objects)):
                obj2 = objects[o2]
                u = (0,0)
                
                if (o1 is not o2)  :

                    rb = obj2.r
                    
                    pb = obj2.curr_pos
                    r1 = ra+rb
                    r2 = float(r1)/tau
                    cen1 = tuple(map(lambda x, y: y - x, pa, pb))
                    cen2 = tuple(cen / float(tau) for cen in cen1)
                    dist = dis(obj1,obj2)
                    dist1 = float(dist/tau)
                    if(dist*dist - (ra+rb)*(ra+rb) < 0): return
                    theta = math.atan((ra+rb)/(math.sqrt(dist*dist - (ra+rb)*(ra+rb))+1e-6))
                    
                    # rotate cen2 by theta angle and make it of unit size
                    xs1 = (cen2[0]*math.cos(theta) - cen2[1]*math.sin(theta))/dist1
                    ys1 = (cen2[1]*math.cos(theta) + cen2[0]*math.sin(theta))/dist1
                        
                    xs2 = (cen2[0]*math.cos(theta) + cen2[1]*math.sin(theta))/dist1
                    ys2 = (cen2[1]*math.cos(theta) - cen2[0]*math.sin(theta))/dist1
                    

                    rel_vel = tuple(map(lambda x, y: y - x, obj2.curr_vel, obj1.curr_vel))
                    rv1 = rel_vel[0]
                    rv2 = rel_vel[1]
                    #projected length of rel_vel on the lines
                    projlen12 = xs1*rv1 + ys1*rv2
                    projlen13 = xs2*rv1 + ys2*rv2
                    #find u if it was towards the lines
                    u12 = (projlen12*xs1- rv1, projlen12*ys1 - rv2)
                    u13 = (projlen13*xs2- rv1, projlen13*ys2 - rv2)
                    

                    if dist > ra+rb:
                    
                        if (cen2[0]*(rv1-cen2[0]) + cen2[1]*(rv2-cen2[1]) <0) and (cen2[0]*(rv1-cen2[0]) + cen2[1]*(rv2-cen2[1]))**2 > (r2**2)*((rv1-cen2[0])**2 + (rv2-cen2[1])**2):
                            u1 = tuple(map(lambda x, y: y - x, cen2, rel_vel))
                            f = float((r2 - math.sqrt(u1[1]*u1[1] + u1[0]*u1[0]))/math.sqrt(u1[1]*u1[1] + u1[0]*u1[0]))/2
                            # print("c1")
                            u11 = tuple(ur2*f for ur2 in u1)
                            u = u11
                            uinreg = tuple(ut*2 for ut in u)
                            sl = float(-1*u[0]/(u[1]+1e-6))
                            if(float(math.sqrt(u1[1]*u1[1] + u1[0]*u1[0])) > float((ra+rb)/tau)):
                                uinreg = (0,0)
                            slope.append(sl)
                            point1 = tuple(map(lambda x, y: y + x, u, obj1.curr_vel))
                            # ponline.append(point1)
                            intcept = point1[1] - sl *(point1[0])
                            intercepts.append(intcept)
                            point2 = tuple(map(lambda x, y: y + x, uinreg, obj1.curr_vel))
                            pinreg.append(point2)
                            # print(u)
                        else:
                            
                            if (rv1*cen2[1] - rv2*cen2[0]) < 0.0:
                                # print("c2")
                                
                                u = u12
                                uinreg = u
                                # u = tuple(ut1*1.2 for ut1 in u)
                                if obj2.opt_speed > 1e-2:
                                    u = tuple(ut1/2 for ut1 in u)
                                        
                                sl = float(ys1/(xs1+1e-6))
                                
                                slope.append(sl)
                                point1 = tuple(map(lambda x, y: y + x, u, obj1.curr_vel))
                                if(rv1*ys1 - rv2*xs1 < 0):
                                    # point1 = (0,0)
                                    uinreg = (0.0,0.0)
                                    # print("c22")
                                ponline.append(point1)
                                intcept = point1[1] - sl *(point1[0])
                                intercepts.append(intcept)
                                point2 = tuple(map(lambda x, y: y + x, uinreg, obj1.curr_vel))
                                pinreg.append(point2)
                            else:
                                # print("c3")
                                
                                u = u13
                                uinreg = u
                                # u = tuple(ut1*1.2 for ut1 in u)
                                if obj2.opt_speed > 1e-2:
                                    u = tuple(ut1/2 for ut1 in u)
                                    
                                sl = float(ys2/(xs2+1e-6))
                                slope.append(sl)
                                point1 = tuple(map(lambda x, y: y + x, u, obj1.curr_vel))
                                if(rv1*ys2 - rv2*xs2 > 0):
                                    # point1 = (0,0)
                                    uinreg = (0.0,0.0)
                                    # print("c4")
                                ponline.append(point1)
                                intcept = point1[1] - sl *(point1[0])
                                intercepts.append(intcept)
                                point2 = tuple(map(lambda x, y: y + x, uinreg, obj1.curr_vel))
                                pinreg.append(point2)
                       
            closest_point = find_closest_point(slope, intercepts, pinreg, opt)
        
            result = np.array(closest_point).flatten()
            vel.append(result)
        for o3 in range(len(objects)):
            obj1 = objects[o3]
            obj1.curr_vel = vel[o3]
            d1 = (1.0, 1.0)
            dcover1 = tuple(x * y for x, y in zip(d1, obj1.curr_vel))
            obj1.curr_pos = tuple(map(lambda x, y: y + x, dcover1, obj1.curr_pos))
    
    



def fhat(xhat):
    n, d = xhat.shape
    objects = []
    for i in range(n):
        obj = Object(
            radius=float(xhat[i][0]),  # Use float instead of int
            current_position=(float(xhat[i][1]), float(xhat[i][2])),  # Use float instead of int
            final_position=(float(xhat[i][3]), float(xhat[i][4])),  # Use float instead of int
            optimal_speed=float(xhat[i][5]),  # Use float instead of int
            current_velocity=(float(xhat[i][6]), float(xhat[i][7]))  # Use float instead of int
        )
        if obj.opt_speed == 0.0:
            obj.curr_vel = (0.0, 0.0)
        objects.append(obj)
    
    moveonestep(objects)
    
    for i in range(n):
        obj = objects[i]
        xhat[i][0] = float(obj.r)
        xhat[i][1] = obj.curr_pos[0]
        xhat[i][2] = obj.curr_pos[1]
        xhat[i][3] = obj.fin_pos[0]
        xhat[i][4] = obj.fin_pos[1]
        xhat[i][5] = obj.opt_speed
        xhat[i][6] = obj.curr_vel[0]
        xhat[i][7] = obj.curr_vel[1]
    
    return xhat
