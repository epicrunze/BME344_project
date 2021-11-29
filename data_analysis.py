import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reject_outliers(data, m = 2., key=None):
    
    relevant_data = [key(datum) for datum in data] if key is not None else data
    d = np.abs(relevant_data - np.median(relevant_data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def main(file_name):
    df = pd.read_csv(file_name)

    df_dict = df.to_dict(orient="list")
    
    data = np.array(list(zip(df_dict["time"], df_dict["x_loc"], df_dict["radius"])))
    data = reject_outliers(data, m = 2., key=lambda x: x[1])
    data = data.T

    mean_size = np.mean(data[2])
    cm_per_px = 2 * 2.54 / mean_size
    print(f"{cm_per_px} is the cm/px")

    fig, (ax1, ax2, ax3) = plt.subplots(3, )
    ax1.plot(data[0], data[1])
    ax1.set(xlabel="time (seconds)", ylabel="Location of Horizontal centroid")

    mean_loc = np.mean(data[1])
    deviations = data[1] - mean_loc

    ax2.plot(data[0], deviations * cm_per_px )
    ax2.set(xlabel="time (seconds)", ylabel="deviation of x val from centroid (cm)")

    angle_data = np.arctan(deviations * cm_per_px / 148)

    ax3.plot(data[0], angle_data)
    ax3.set(xlabel="time (seconds)", ylabel="angle of body (rad)")

    fig.set_size_inches(18.5, 10.5)
    plt.show()

    df = pd.DataFrame({
        "time": data[0],
        "angular_deviation": angle_data,
    })

    df.to_csv("Angular_Dev_and_Time.csv", index=False)


if __name__ == "__main__":
    main(file_name = "output.csv")