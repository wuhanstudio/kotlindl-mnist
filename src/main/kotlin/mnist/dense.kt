package mnist

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.mnist

fun denseMnist() {
    val (train, test) = mnist()

    val model = Sequential.of(
        Input(28, 28, 1),
        Flatten(),
        Dense(256),
        Dense(128),
        Dense(10, activation = Activations.Softmax)
    )

    model.use{
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val start = System.currentTimeMillis()
        it.fit(dataset = train, epochs = 10)
        println("Training time: ${(System.currentTimeMillis() - start) / 1000f}")

        val accuracy = it.evaluate(test)

        println(accuracy)
    }
}
