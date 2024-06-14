import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
from matplotlib.patches import Polygon
from decimal import Decimal
import copy
import matplotlib.collections as mcoll

import seaborn as sns
sns.set_style("darkgrid")

def get_action_points(action_times, action, current_time):
    last_idx = (action_times < current_time)*1
    last_idx = last_idx.sum()
    plot_points = []
    plot_points.append([action_times[0], action[0]])
    for i in range(1, last_idx):
        plot_points.append([action_times[i], action[i-1]])
        plot_points.append([action_times[i], action[i]])

    plot_points.append([current_time, action[last_idx-1]])

    return np.array(plot_points)

class HydraylicAnimation():
    def __init__(
        self, 
        trajectory_df, 
        actions_df, 
        l_critic, 
        d_drop_ration, 
        n_frames=100, 
        figsize=(12, 6)
    ):

        plt.rcParams.update({'font.size': 10,
                             'grid.color': 'gray',
                             'axes.linewidth': 1.5,
                             'animation.embed_limit': 40.0,
                             'legend.framealpha': 0.8,
                             'legend.frameon': True,
                             'axes.edgecolor': 'grey'})
        
        self.figsize = figsize

        self.pst_w = 5          # piston width
        self.res_down_c = 7     # upper piston offset
        self.res_hight = 18     # piston high
        self.pst_thick = 2      # piston thickness
        self.pst_rog = 1        # piston rog diameter
        self.res_thick = 0.5    # reseuares thickness
        self.nozle_d = 0.8      # nozle diameter
        self.nozle_h = 1        # nozle hight
        self.res_down_offset = 1    # down reseuar offset from max piston position 
        self.res_up_offset = 1      # upper reseuar offest from max piston position    
        self.res_up_hole = 1.5      # hole diameter in upper reseuar
        self.res_up_h = 6           # length of upper reseuar 
        self.piston_amlitude = 1.5  # piston strike     

        self.d_drop_ration = d_drop_ration # nozzle/drop_diameter ration

        self.vent_w = 2     # throttle width
        self.vent_t = 0.5   # throttle thickness
        self.vent_r = 0.6   # throttle rog diameter
        self.vent_h = 3     # throttle height

        # to get gradient in nozzle it is splitted in height to equal small polygons
        self.nozzle_coords = []
        self.n_grad_steps = 10 # number of subpolygones
        self.nozzle_y_coords = np.linspace(-self.nozle_h-self.res_thick, 0, self.n_grad_steps)

        for i in range(self.nozzle_y_coords.shape[0]-1):
            self.nozzle_coords.append(
                np.array([
                    [-self.nozle_d/2, self.nozzle_y_coords[i]],
                    [-self.nozle_d/2, self.nozzle_y_coords[i+1]],
                    [self.nozle_d/2, self.nozzle_y_coords[i+1]],
                    [self.nozle_d/2, self.nozzle_y_coords[i]]
                    ])
            )

        # throttle coordinates (it is moved by y offset)
        self.vent_coords = np.array([
            [-self.vent_w/2, self.res_hight-self.vent_t],
            [-self.vent_w/2, self.res_hight],
            [-self.vent_r/2, self.res_hight],
            [-self.vent_r/2, self.res_hight+self.vent_h],
            [self.vent_r/2, self.res_hight+self.vent_h],
            [self.vent_r/2, self.res_hight],
            [self.vent_w/2, self.res_hight],
            [self.vent_w/2, self.res_hight-self.vent_t]
        ])

        # coordinates of reseuares polygones
        self.reservars_coords = [
            np.array([
                [-self.nozle_d/2, 0],
                [-self.nozle_d/2, -self.nozle_h-self.res_thick],
                [-self.nozle_d/2-self.res_thick, -self.nozle_h-self.res_thick],
                [-self.nozle_d/2-self.res_thick, -self.res_thick],
                [-self.pst_w/2-self.res_thick, -self.res_thick],
                [-self.pst_w/2-self.res_thick, self.res_down_c+self.res_down_offset],
                [-self.pst_w/2, self.res_down_c+self.res_down_offset],
                [-self.pst_w/2, 0],
            ]),
            np.array([
                [self.nozle_d/2, 0],
                [self.nozle_d/2, -self.nozle_h-self.res_thick],
                [self.nozle_d/2+self.res_thick, -self.nozle_h-self.res_thick],
                [self.nozle_d/2+self.res_thick, -self.res_thick],
                [self.pst_w/2+self.res_thick, -self.res_thick],
                [self.pst_w/2+self.res_thick, self.res_down_c+self.res_down_offset],
                [self.pst_w/2, self.res_down_c+self.res_down_offset],
                [self.pst_w/2, 0],
            ]),
            np.array([
                [-self.res_up_hole/2, self.res_hight],
                [-self.pst_w/2, self.res_hight],
                [-self.pst_w/2, self.res_hight-self.res_up_h],
                [-self.pst_w/2-self.res_thick, self.res_hight-self.res_up_h],
                [-self.pst_w/2-self.res_thick, self.res_hight+self.res_thick],
                [-self.res_up_hole/2, self.res_hight+self.res_thick],
            ]),
            np.array([
                [self.res_up_hole/2, self.res_hight],
                [self.pst_w/2, self.res_hight],
                [self.pst_w/2, self.res_hight-self.res_up_h],
                [self.pst_w/2+self.res_thick, self.res_hight-self.res_up_h],
                [self.pst_w/2+self.res_thick, self.res_hight+self.res_thick],
                [self.res_up_hole/2, self.res_hight+self.res_thick],
            ])
        ]

        # trajectories parsing
        # sparse trajectories
        trajectory_df = trajectory_df[trajectory_df.index % int(trajectory_df.shape[0]/n_frames) == 0]

        # parsing actions dataset        
        self.actions = actions_df['throttle action [µm]'].to_numpy()
        self.time_action = actions_df['time'].to_numpy()

        # parsing observation dataset
        self.obs_jet_length = actions_df['jet length [mm]'].to_numpy()
        self.obs_jet_vel = actions_df['jet velocity [mm/s]'].to_numpy()

        # parsing dynamic dataset
        # parsing time
        self.time = trajectory_df['time'].to_numpy()

        # parse, shift and scale piston position array
        self.x_p_vec_init = trajectory_df['piston position [µm]'].to_numpy()
        self.x_p_vec = self.x_p_vec_init - 1000                              
        self.x_p_vec = 1.5 + self.piston_amlitude*self.x_p_vec/self.x_p_vec.max()

        # parse, scale and shift pressures arrays
        self.p_up_vec_init = trajectory_df['hydraulic pressure [Pa]'].to_numpy()/1000
        self.p_down_vec_init = trajectory_df['working pressure [Pa]'].to_numpy()/1000
        self.p_up_vec = self.p_up_vec_init - self.p_up_vec_init.min()       # is needed for coloring
        self.p_down_vec = self.p_down_vec_init - self.p_down_vec_init.min()

        # parse and scale and cliping (no less then zero) throttle position
        self.x_th_vec_init = trajectory_df['throttle position [µm]'].to_numpy() 
        self.x_th_vec_init = np.clip(self.x_th_vec_init, 0, self.x_th_vec_init.max())
        self.x_th_vec = 0.8*self.x_th_vec_init/self.x_th_vec_init.max()
        self.x_th_vec = np.clip(self.x_th_vec, 0, self.x_th_vec.max())

        # parse jet length array
        self.l_jet_vec_init = trajectory_df['jet length [mm]'].to_numpy()
        self.l_jet_vec = self.l_jet_vec_init

        # parse jet velocity
        self.v_jet_vec_init = trajectory_df['jet velocity [mm/s]'].to_numpy()
        self.v_jet_vec = self.v_jet_vec_init
        self.l_critic = l_critic

        # create colormaps for pressures coloring
        self.p_down_min = self.p_down_vec.min()         # used to difive a gradient in nozzle
        self.min_p_down_id = np.argmin(self.p_down_vec) 
        self.cmap = matplotlib.colormaps['winter']

        # scaling arrays to get convinient colormap for both reseuares
        self.cm_offset = 0
        self.pressure_limit = min(self.p_down_vec.max(), self.p_up_vec.max())
        self.pressure_max = self.pressure_limit * 1.5
        self.p_down_scaled = self.p_down_vec/(self.pressure_max+self.cm_offset)
        self.p_up_scaled = self.p_up_vec/(self.pressure_max+self.cm_offset) 

        self.down_colors = self.cmap(self.p_down_scaled)             # colormap for down reseuar   
        self.up_colors = self.cmap(self.p_up_scaled)                 # colormap for upper reseuar   
        self.min_p_down_color = self.down_colors[self.min_p_down_id] # is needed to drow jet and drops

        # config for plots
        self.plot_dict = {3: {'data': self.l_jet_vec_init,
                              'color': 'red',
                              'y_tit': 'jet length [mm]',
                              'label1': 'ground truth',
                              'label2': 'noise observation',
                              'legend_loc': 'center right',
                              'observation': self.obs_jet_length,
                              'h_line': l_critic},
                    0: {'data': self.p_down_vec_init, 'data2': self.p_up_vec_init,
                         'color': 'red', 'color2': 'teal',
                         'label1': 'working pressure [kPa]',
                         'label2': 'hydraulic pressure [kPa]',
                         'legend_loc': 'center right',
                           'y_tit': 'hydraulic/working pressure, [kPa]'},
                    2: {'data': self.x_th_vec_init,
                        'data2': None,
                        'actions': True,
                        'label1': 'throttle position [µm]',
                        'label2': 'throttle action [µm]',
                        'legend_loc': 'upper right',
                        'color': 'green', 'color2': 'red',
                        'h_line': 0,
                        'y_tit': 'throttle position [µm]'},
                    4: {'data': self.v_jet_vec_init,
                         'color': 'blue',
                         'y_tit': 'jet velocity [mm/s]',
                         'label1': 'ground truth',
                         'label2': 'noise observation',
                         'legend_loc': 'upper right',
                         'observation': self.obs_jet_vel}
                    }

        # difine statics plot limits for plots
        for plot_name in self.plot_dict:
            self.plot_dict[plot_name]['x_max'] = self.time.max()
            if 'data2' in self.plot_dict[plot_name] and 'actions' not in self.plot_dict[plot_name]:
                y_min = min(self.plot_dict[plot_name]['data'].min(), self.plot_dict[plot_name]['data2'].min())
                y_max = max(self.plot_dict[plot_name]['data'].max(), self.plot_dict[plot_name]['data2'].max())

            elif 'actions' in self.plot_dict[plot_name]:
                y_min = min(self.plot_dict[plot_name]['data'].min(), self.actions.min())
                y_max = max(self.plot_dict[plot_name]['data'].max(), self.actions.max())

            else:
                y_min = self.plot_dict[plot_name]['data'].min()
                y_max = self.plot_dict[plot_name]['data'].max()

            hight = y_max - y_min
            y_min -= 0.05*hight
            y_max += 0.05*hight

            self.plot_dict[plot_name]['y_min'] = y_min
            self.plot_dict[plot_name]['y_max'] = y_max

        # to storage drop info
        self.drops_time = []
        self.drops_init_vel = []
        self.time_clock_drop = 0

    def init_plot(self):
        # create subplots and fixed subplot's positions
        self.fig, self.axes = plt.subplots(1, 5, frameon=False, figsize=self.figsize)
        self.fig.patch.set_alpha(1)
        

        self.axes[0].set_position([0.05, 0.07, 0.3, 0.4])
        self.axes[1].set_position([0.4, 0.05, 0.2, 0.9])
        self.axes[2].set_position([0.05, 0.55, 0.3, 0.4])
        self.axes[3].set_position([0.65, 0.07, 0.3, 0.4])
        self.axes[4].set_position([0.65, 0.55, 0.3, 0.4])

    def get_circles(self, diam=0.3):
        """
        The function returns the circles of drops.
        """
        # get drop coordinates
        coords = [v*t + 9800*(t**2)/2 for v, t in zip(self.drops_init_vel, self.drops_time)]
        y0 = -self.nozle_h-self.res_thick-self.l_critic
        true_coords = [y0 - y for y in coords]

        return [plt.Circle((0, y), diam/2, color=self.min_p_down_color) for y in true_coords]

    def get_polygon(self, x_p, res_type, l_jet=None, x_th=None, p_down=None, c_up=None, c_down=None):
        """
        The function returns the polygones.
        """
        # down reseurar polygone
        if res_type == 'down':
            polygon_coords = np.array([
                [-self.pst_w/2,0],
                [-self.pst_w/2, self.res_down_c-x_p],
                [self.pst_w/2, self.res_down_c-x_p],
                [self.pst_w/2,0],
                ])
            return Polygon(polygon_coords, facecolor = c_down, edgecolor=c_down)
        
        # upper reseuar polygone
        elif res_type == 'up':
            polygon_coords = np.array([
                [-self.pst_w/2, self.res_hight-x_p],
                [-self.pst_w/2, self.res_hight],
                [self.pst_w/2, self.res_hight],
                [self.pst_w/2, self.res_hight-x_p]
                ])
            
            return Polygon(polygon_coords, facecolor = c_up, edgecolor=c_up)
        
        # nozzle polygones
        elif res_type == 'nozle':
            # get colormap for polygones in nozzle
            nozle_p_vec = np.linspace(self.p_down_min, p_down, self.n_grad_steps)
            nozle_colors = self.cmap(nozle_p_vec/(self.pressure_max+self.cm_offset))

            polygons = []
            for j, polygon_coords in enumerate(self.nozzle_coords):
                polygons.append(Polygon(polygon_coords, facecolor = nozle_colors[j], edgecolor=nozle_colors[j]))

            return polygons
        
        # jet polygone
        elif res_type == 'jet':
            polygon_coords = np.array([
                [-self.nozle_d/2, -self.nozle_h-self.res_thick-l_jet],
                [-self.nozle_d/2, -self.nozle_h-self.res_thick],
                [self.nozle_d/2, -self.nozle_h-self.res_thick],
                [self.nozle_d/2, -self.nozle_h-self.res_thick-l_jet]
                ])
            
            return Polygon(polygon_coords, facecolor = self.min_p_down_color, edgecolor=self.min_p_down_color)
        
        # throttle polygone
        elif res_type == 'vent':
            polygon_coords = np.copy(self.vent_coords)
            polygon_coords[:, 1] = polygon_coords[:, 1] - x_th

            return Polygon(polygon_coords, facecolor = 'black', edgecolor='black')
    
        # piston polygone
        elif res_type == 'piston':
            polygon_coords = np.array([
                [-self.pst_w/2, self.res_down_c-x_p],
                [-self.pst_w/2, self.res_down_c-x_p+self.pst_thick],
                [-self.pst_rog/2, self.res_down_c-x_p+self.pst_thick],

                [-self.pst_rog/2, self.res_hight-x_p-self.pst_thick],
                [-self.pst_w/2, self.res_hight-x_p-self.pst_thick],
                [-self.pst_w/2, self.res_hight-x_p],

                [self.pst_w/2, self.res_hight-x_p],
                [self.pst_w/2, self.res_hight-x_p-self.pst_thick],
                [self.pst_rog/2, self.res_hight-x_p-self.pst_thick],

                [self.pst_rog/2, self.res_down_c-x_p+self.pst_thick],
                [self.pst_w/2, self.res_down_c-x_p+self.pst_thick],
                [self.pst_w/2, self.res_down_c-x_p]
                ])
            
            return Polygon(polygon_coords, facecolor = 'k', edgecolor='k')
        
        # static parts of reseuars polygones
        elif res_type == 'reservars':
            polygons = [Polygon(polygon_coords, facecolor = 'dimgray', edgecolor='dimgray') for polygon_coords in self.reservars_coords]
            return polygons
        
        # background polygone 
        elif res_type == 'background':
            polygon_coords = np.array([
            [-self.pst_w/2-10, -10],
            [-self.pst_w/2-10, self.res_hight + 2],
            [self.pst_w/2+10, self.res_hight + 2],
            [self.pst_w/2+10, -10],
            ])
            return Polygon(polygon_coords, facecolor = 'white')

    def init(self):
        # is needed for few iterations
        for j, ax in enumerate(self.axes):
            ax.clear()

        self.axes[1].axis('off')
        self.axes[1].set(frame_on=False)
        self.axes[1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
        
        self.axes[1].clear()
        self.axes[1].set_xlim([-self.pst_w/2 - 2, self.pst_w/2 + 2])
        self.axes[1].set_ylim([ - 10, self.res_hight + 2])
        
        self.drops_time = []
        self.drops_init_vel = []

        return self.axes

    def animate(self, i):

        self.time_units = 1000 # ms
        
        for j, ax in enumerate(self.axes):
            
            ax.clear()

            if j != 1: # the 1th axis is used form plotting system
                # implement titles, limits, grids, tickets
                self.axes[j].set_xlim([0, self.plot_dict[j]['x_max']*self.time_units])
                self.axes[j].set_ylim([self.plot_dict[j]['y_min'], self.plot_dict[j]['y_max']])
                self.axes[j].set_xlabel(xlabel='Time, [ms]', fontsize=10)
                self.axes[j].set_ylabel(self.plot_dict[j]['y_tit'], fontsize=10)

                self.axes[j].grid(which = "major", linewidth = 1)
                self.axes[j].grid(which = "minor", linewidth = 0.2)
                self.axes[j].minorticks_on()

                if 'data2' in self.plot_dict[j]: # for the axes with 2 curve
                    self.axes[j].plot(self.time[:i]*self.time_units, self.plot_dict[j]['data'][:i], c=self.plot_dict[j]['color'], label=self.plot_dict[j]['label1'])
                
                    if 'actions' in self.plot_dict[j]: # for the action curve (it has anther time array)
                        if i > 0:
                            action_data = get_action_points(self.time_action, self.actions, self.time[i])
                            self.axes[j].plot(action_data[:, 0]*self.time_units, action_data[:, 1], # to ms 
                                              c=self.plot_dict[j]['color2'], 
                                              label=self.plot_dict[j]['label2'], linewidth = 2.5, linestyle = '--')
                    else:
                        self.axes[j].plot(self.time[:i]*self.time_units, self.plot_dict[j]['data2'][:i],
                                           c=self.plot_dict[j]['color2'], label=self.plot_dict[j]['label2'])
                    # legend only for plots with 2 curve
                    self.axes[j].legend(loc=self.plot_dict[j]['legend_loc'], fontsize=9)

                else:
                    if 'label1' in  self.plot_dict[j]:
                        self.axes[j].plot(self.time[:i]*self.time_units, 
                                          self.plot_dict[j]['data'][:i], 
                                          c=self.plot_dict[j]['color'], 
                                          label=self.plot_dict[j]['label1'])
                    else:  
                        self.axes[j].plot(self.time[:i]*self.time_units, self.plot_dict[j]['data'][:i], c=self.plot_dict[j]['color'])

                if 'observation' in self.plot_dict[j]:
                    current_ids = (self.time_action < self.time[i])
                    self.axes[j].scatter(self.time_action[current_ids]*self.time_units,
                                         self.plot_dict[j]['observation'][current_ids],
                                         label=self.plot_dict[j]['label2'],
                                         c=self.plot_dict[j]['color'])
                    self.axes[j].legend(loc=self.plot_dict[j]['legend_loc'], fontsize=9) 

                # add horizontal line for some plots
                if 'h_line' in self.plot_dict[j]:
                    self.axes[j].axhline(y = self.plot_dict[j]['h_line'], color = 'black', linewidth = 2.0, linestyle = '--') 
        
        # axis config for 1th axis
        self.axes[1].axis('equal')
        self.axes[1].set_xlim([-self.pst_w/2 - 2, self.pst_w/2 + 2])
        self.axes[1].set_ylim([ - 10, self.res_hight + 2])
        self.axes[1].axis('off')
        
        # get polygon up
        polygon_up = self.get_polygon(self.x_p_vec[i], 'up', c_up=self.up_colors[i])

        # get polygon down
        polygon_down = self.get_polygon(self.x_p_vec[i], 'down', c_down=self.down_colors[i])

        # get piston polygon
        polygon_piston = self.get_polygon(self.x_p_vec[i], 'piston')

        # get static polygons
        polygons_res = self.get_polygon(self.x_p_vec[i], 'reservars')

        # get background polygon
        polygon_back = self.get_polygon(self.x_p_vec[i], 'background')

        # get vent polygon
        polygon_vent = self.get_polygon(self.x_p_vec[i], 'vent', x_th=self.x_th_vec[i])

        # get nozle polygon 
        polygons_nozle = self.get_polygon(self.x_p_vec[i], 'nozle', p_down=self.p_down_vec[i], c_down=self.down_colors[i])

        # update time for drops if it exist
        if len(self.drops_time) > 0:
            d_t = self.time[i] - self.time_clock_drop 
            self.drops_time = [t+d_t for t in self.drops_time]
            self.time_clock_drop = self.time[i]

        # create drop
        if self.l_jet_vec[i] > self.l_critic:
            
            l_jet = self.l_jet_vec[i]
            i_drop_c = 1
            while l_jet > self.l_critic:
                l_jet = l_jet - self.l_critic

                # add only if it is new drop
                if i_drop_c > len(self.drops_time):
                    self.drops_time.append(0)
                    self.time_clock_drop = self.time[i]
                    self.drops_init_vel.append(self.v_jet_vec[i])
                i_drop_c += 1            
        else:
            l_jet = self.l_jet_vec[i]


        # get jet polygon 
        if l_jet > 0:
            polygon_jet = self.get_polygon(self.x_p_vec[i], 'jet', l_jet=l_jet)

        # drop drowing
        # get time of droping
        if len(self.drops_time) > 0:
            # get circles of drops
            circles_drops = self.get_circles(self.d_drop_ration*self.nozle_d)
            
        # implement polygones to axis
        self.axes[1].add_patch(polygon_back)

        for stat_polygon in polygons_res:
            self.axes[1].add_patch(stat_polygon)

        for nozle_polygon in polygons_nozle:
            self.axes[1].add_patch(nozle_polygon)

        if l_jet > 0:
            self.axes[1].add_patch(polygon_jet)

        if len(self.drops_time) > 0:
            for crcl_drop in circles_drops:
                self.axes[1].add_patch(crcl_drop)

        self.axes[1].add_patch(polygon_up)
        self.axes[1].add_patch(polygon_down)
        self.axes[1].add_patch(polygon_piston)
        self.axes[1].add_patch(polygon_vent)

        return self.axes


    def animate_system(self, interacitve=False, save_gif = False, gif_filename = 'pistons.gif'):
                
        self.init_plot()
        plt.close()
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                    frames=self.time.shape[0], interval=20) #, blit=True)

        if save_gif:
            # # To save the animation using Pillow as a gif
            writer = animation.PillowWriter(fps=40,
                                            codec="gif",
                                            metadata=dict(artist='Me'),
                                            bitrate=5000)
            
            anim.save(gif_filename, writer=writer, dpi=300, savefig_kwargs={"transparent": False, 'facecolor':'red'})

        if interacitve: # interactive mode
            from matplotlib import rc
            rc('animation', html='jshtml')
            return anim
        
        else: # to show animation in window
            plt.show()

    def save_last_frame(self, filename):
        self.init_plot()
        for i in range(self.time.shape[0]): ## needed cycle to collect time for drop dynamic
            axes = self.animate(i)
            if i == self.time.shape[0]-1:
                self.fig.savefig(filename, bbox_inches='tight')
        plt.close()

    def show_last_frame(self):
        self.init_plot()
        for i in range(self.time.shape[0]): ## needed cycle to collect time for drop dynamic
            axes = self.animate(i)
            if i == self.time.shape[0]-1:
                break

        plt.show()
