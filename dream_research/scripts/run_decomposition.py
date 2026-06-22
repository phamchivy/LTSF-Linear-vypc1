import torch
import torch.nn as nn

from data_provider.data_factory import data_provider

from dream_research.decomposition_analysis.benchmark_decomposition import DecompositionBenchmark

from dream_research.models.decomposition.ma_residual import MAResidualDecomposition
# from dream_research.models.decomposition.double_ma import DoubleMADecomposition
# from dream_research.models.decomposition.multiscale_conv import MultiScaleConvDecomposition


def run(args):

    test_set, test_loader = data_provider(args, flag='test')

    decomposition_dict = {
        "MA": MAResidualDecomposition(kernel_size=25),
        # "DoubleMA": DoubleMADecomposition(),
        # "Conv": MultiScaleConvDecomposition()
    }

    criterion = nn.L1Loss()

    for name, decomposition in decomposition_dict.items():

        bench = DecompositionBenchmark(
            decomposition,
            args,
            criterion
        )

        total_mae = 0
        total_energy = 0
        count = 0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:

            batch_x = batch_x.float()
            batch_y = batch_y.float()

            metrics = bench.evaluate(
                batch_x,
                batch_y[:, -args.pred_len:, :]
            )

            total_mae += metrics["residual_mae"]
            total_energy += metrics["residual_energy"]

            count += 1

        print("=" * 50)
        print(name)
        print(
            "Residual MAE:",
            total_mae / count
        )

        print(
            "Residual Energy:",
            total_energy / count
        )


if __name__ == "__main__":

    from run_longExp import args

    run(args)