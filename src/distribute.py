# coding: utf-8
# distribute.py
# to accelerate the tf train rate with distributed running.

import tensorflow as tf
import cnnm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

if FLAGS.job_name == 'ps':
	server.join()
elif FLAGS.job_name == 'worker':
	with tf.device(tf.train.replica_device_setter(
		worker_device = '/job:worker/task:%d' % FLAGS.task_index,
		cluster = cluster)):

		# network describe code here.

		train_step = optimizer.minimize(a.cross_entropy)
		init_op = tf.initialize_all_variables()
		saver = tf.train.Saver()

		tf.scalar_summary('cost', a.cross_entropy)
		summary_op = tf.merge_all_summaries()
		sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
						logdir="./checkpoint/",
						init_op=init_op,
						summary_op=summary_op,
						saver=saver,
						global_step=global_step,
						save_model_secs=60)

		with sv.prepare_or_wait_for_session(server.target) as sess:
			steps = 10000
			batch_size = 50
			print 'training %d steps with batch size %d' %(steps, batch_size)
			for i in range(steps):
				batch = b.next_batch(batch_size)
				if i%10 == 0:
					train_accuracy = a.accuracy.eval(feed_dict={a.x:batch[0], a.y:batch[1]})
					print 'step %d, train accuracy %g' %(i, train_accuracy)
				sess.run([train_step], feed_dict={a.x:batch[0], a.y:batch[1]})
		sv.stop()

if __name__ == '__main__':
	tf.app.run()