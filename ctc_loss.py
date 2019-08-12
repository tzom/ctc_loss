import tensorflow as tf
import numpy as np 

tf.enable_eager_execution()

batch_size=1


predictions_len = np.repeat(30,batch_size)
targets_len     = np.repeat(22+1,batch_size)


#predictions = np.random.uniform(-10,10,size=(batch_size,30,22+1)).astype(np.float32)
predictions = np.identity(22+1).reshape(1,22+1,22+1).astype(np.float32)
predictions = tf.transpose(predictions,perm=[1,0,2])
targets = np.array(range(22))
targets = targets.reshape(1,-1)

loss = tf.nn.ctc_loss_v2(targets,predictions,label_length=targets_len,logit_length=predictions_len,logits_time_major=True)

loss = tf.reduce_mean(loss)

print(loss)

decoded,probs = tf.nn.ctc_beam_search_decoder_v2(predictions,sequence_length=[23],beam_width=3,top_paths=3)
#decoded,probs = tf.nn.ctc_beam_search_decoder(predictions,sequence_length=targets_len+1,beam_width=5,top_paths=5,merge_repeated=True)

print(probs)
print(tf.sparse.to_dense(decoded[0]))
print(tf.sparse.to_dense(decoded[1]))
print(tf.sparse.to_dense(decoded[2]))

decoded,probs = tf.nn.ctc_greedy_decoder(predictions,sequence_length=[23],merge_repeated=True)
print(tf.sparse.to_dense(decoded[0]))
print(probs)

#print(tf.math.argmax(predictions))