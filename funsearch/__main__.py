import json
import logging
# logging.getLogger("httpx").disabled = True
import os
import pathlib
import pickle
import time

import click
# import llm
from transformers import BitsAndBytesConfig
import torch.multiprocessing as mp
# import ollama
from dotenv import load_dotenv
# import debugpy
import numpy as np

# # Allow other machines to attach to debugpy at port 5678
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # Pause the program until a debugger attaches
# print("Debugger attached. Continuing execution.")


from funsearch import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator, custom_sampler

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
mp.set_start_method('spawn', force=True)

def get_all_subclasses(cls):
  all_subclasses = []

  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(get_all_subclasses(subclass))

  return all_subclasses


SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


def parse_input(filename_or_data: str):
  if len(filename_or_data) == 0:
    raise Exception("No input data specified")
  p = pathlib.Path(filename_or_data)
  if p.exists():
    if p.name.endswith(".json"):
      return json.load(open(filename_or_data, "r"))
    if p.name.endswith(".pickle"):
      return pickle.load(open(filename_or_data, "rb"))
    raise Exception("Unknown file format or filename")
  if "," not in filename_or_data:
    data = [filename_or_data]
  else:
    data = filename_or_data.split(",")
  if data[0].isnumeric():
    f = int if data[0].isdecimal() else float
    data = [f(v) for v in data]
  return data

# defining this at module level
def create_evaluator(database, sandbox_class, template, function_to_evolve, 
                    function_to_run, inputs, parametric_program, optimize_floats, 
                    optimization_budget, significant_digits, log_path, spec_filename):
  return evaluator.Evaluator(
      database,
      sandbox_class(base_path=log_path),
      template,
      function_to_evolve,
      function_to_run,
      inputs,
      parametric_program=parametric_program,
      optimize_floats=optimize_floats,
      optimization_budget=optimization_budget,
      significant_digits=significant_digits,
      spec_filename=spec_filename
  )

@click.group()
@click.pass_context
def main(ctx):
  pass


@main.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument('inputs')
# @click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model')
# @click.option('--model_name', default="bigcode/starcoder2-15b-instruct-v0.1", help='LLM model')
@click.option('--model_name', default="Qwen/Qwen2.5-Coder-7B-Instruct", help='LLM model')


# @click.option('--model_name', default="Qwen/Qwen3-1.7B", help='LLM model')


# @click.option('--model_name', default="openai/gpt-oss-20b", help='LLM model')

# @click.option('--model_name', default="unsloth/gpt-oss-20b-bnb-4bit", help='LLM model')


# model_name = "bigcode/starcoder2-15b-instruct-v0.1"
# model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
# Qwen2.5-Coder-7B-Instruct does not have a padding token.
# Asking to pad but the tokenizer does not have a padding token. 
# Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.

# @click.option('--model_name', default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help='LLM model')
# @click.option('--model_name', default="starcoder2:3b", help='LLM model')
# @click.option('--model_name', default="starcoder-carlo", help='LLM model')

@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=1, type=click.INT, help='Samplers')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
@click.option('--optimize_floats', default=False, type=click.BOOL, help='whether to optimize program floats in the loop') # deprecated
@click.option('--parametric_program', default=True, type=click.BOOL, help='whether to use parametric program')
@click.option('--optimization_budget', default=100, type=click.INT, help='Number of evaluations for float optimization')
@click.option('--significant_digits', default=4, type=click.INT, help='Number of significant digits for float values')

def run(spec_file, 
        inputs, 
        model_name, 
        output_path, 
        load_backup, 
        iterations, 
        samplers, 
        sandbox_type, 
        optimize_floats, 
        parametric_program, 
        optimization_budget, 
        significant_digits):
  """ Execute function-search algorithm:

\b
  SPEC_FILE is a python module that provides the basis of the LLM prompt as
            well as the evaluation metric.
            See examples/cap_set_spec.py for an example.\n
\b
  INPUTS    input filename ending in .json or .pickle, or a comma-separated
            input data. The files are expected contain a list with at least
            one element. Elements shall be **passed to the solve() method**
            one by one. Examples
              8
              8,9,10
              ./examples/cap_set_input_data.json
            [carlo] in the cap set case it is an int representing the 
              dimension of the space where we are looking for a cap set
"""

  import logging
  logging = logging.getLogger(__name__)  # assign before use
  load_dotenv()

  timestamp = str(int(time.time()))
  folder_name = f"{timestamp}"
  
    # Task log
  for k in ["cheetah", "ballcup", "hopper", "reacher", "double_swingup", "double", "fish", "finger", "quadcopter", "quadruped", "unitree"]:
      if k in spec_file.name:
          folder_name = f"{k}_{inputs}_{folder_name}"
          sim_env = k
          break
 

  folder_name = f"{model_name.split('/')[1]}_{folder_name}"

  conf = config.Config()

  import wandb

  # import pdb; pdb.set_trace()


  # Start a new wandb run to track this script.
  wandb_run = wandb.init(
      # Set the wandb entity where your project will be logged (generally your team name).
      entity="design_automation",
      # Set the wandb project where this run will be logged.
      project="polaris",

      # Track hyperparameters and run metadata.
      config={
          "sim_env" : sim_env,
          "model_name": model_name,
          "consumer_producer": True,
          "num_samplers": conf.num_samplers, 
          "num_evaluators": conf.num_evaluators, 
          "samples_per_prompt": conf.samples_per_prompt,  
          "seed": conf.seed, 

          # ProgramsDatabaseConfig entries
          "functions_per_prompt": conf.programs_database.functions_per_prompt,
          "num_islands": conf.programs_database.num_islands,
          "reset_period": conf.programs_database.reset_period,
          "cluster_sampling_temperature_init": conf.programs_database.cluster_sampling_temperature_init,
          "cluster_sampling_temperature_period": conf.programs_database.cluster_sampling_temperature_period,
          "backup_period": conf.programs_database.backup_period,
          "backup_folder": conf.programs_database.backup_folder,
          "score_threshold": conf.programs_database.score_threshold,
          "min_score": conf.programs_database.min_score,
      },
  )

  log_path = pathlib.Path(output_path) / folder_name
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")

  specification = spec_file.read()
  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)

  np.random.seed(conf.seed)
  database = programs_database.ProgramsDatabase(
    conf.programs_database, template, function_to_evolve, identifier=timestamp, log_path=log_path, wandb_run = wandb_run)
  if load_backup:
    database.load(load_backup)

  inputs = parse_input(inputs)

  sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
  assert not optimize_floats, "External float optimization is deprecated"
  logging.info(f"Initializing Evaluator with parametric_program: {parametric_program}")
  initial_evaluator = evaluator.Evaluator(database,
                                          sandbox_class(base_path=log_path),
                                          template,
                                          function_to_evolve,
                                          function_to_run,
                                          inputs,
                                          optimize_floats=optimize_floats,
                                          parametric_program=parametric_program, 
                                          optimization_budget=optimization_budget,
                                          significant_digits=significant_digits,
                                          spec_filename=spec_file.name)

  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  initial_evaluator.analyse(initial, island_id=None, version_generated=None, num_llm_inferences=0)


  assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                   "See e.g. the error files under sandbox data.")

  ### ORIGINAL #########################################
  # model = llm.get_model(model_name)
  # lm = sampler.LLM(samples_per_prompt=2, model=model, log_path=log_path)
  # samplers = [sampler.Sampler(database, evaluators, lm)
  #             for _ in range(samplers)]

  # core.run(samplers, database, iterations)
  ######################################################

  ### PARALLELIZED (CUSTOM) ############################
  # Create a partial function with all the fixed arguments
  from functools import partial
  evaluator_factory = partial(
      create_evaluator,
      database=database,
      sandbox_class=sandbox_class,
      template=template,
      function_to_evolve=function_to_evolve,
      function_to_run=function_to_run,
      inputs=inputs,
      parametric_program=parametric_program,
      optimize_floats=optimize_floats,
      optimization_budget=optimization_budget,
      significant_digits=significant_digits,
      log_path=log_path,
      spec_filename=spec_file.name
  )

  samplers = [custom_sampler.CustomSampler(rank=i, 
                                           seed=conf.seed,
                                           model_name=model_name, 
                                           database=database, 
                                           evaluator_factory=evaluator_factory,
                                           num_evaluators=conf.num_evaluators, 
                                           samples_per_prompt=conf.samples_per_prompt,
                                           quantization_config=None,
                                           log_path=log_path) for i in range(conf.num_samplers)]

  logging.info(f"Model name: {model_name}")
  logging.info(f"Number of samplers: [{len(samplers)}]")

  # only a sampler
  # iterations = 20
  # core.run_parallel(samplers, database, iterations)
  samplers[0].initialize_llm()
  # core.run(samplers, database, iterations)

  s = samplers[0]

  import threading
  from queue import Queue, Empty

  q = Queue(maxsize=500)
  should_stop = threading.Event()

  def producer(s, iterations, logging):
      try:
          while iterations != 0 and not should_stop.is_set():
              dict_prompt_sample = s.sample()  # GPU-intensive
              q.put(dict_prompt_sample, block=True, timeout=None)
              logging.info(f"Producer Queue length: {len(dict_prompt_sample['samples'])}")
              logging.info("Producer Queue length: %d", q.qsize())

              if iterations > 0:
                  iterations -= 1
          print("Producer Done")
      except Exception as e:
          logging.exception("Producer exception: %s", e)
          should_stop.set()
      finally:
          print("Producer Done")

          # Sentinel value for consumer to stop
          q.put(None)

  def consumer(s, logging):
      try:
          while not should_stop.is_set():
              dict_prompt_sample = q.get(block=True)

              if dict_prompt_sample is None:
                  break
              logging.info("Consumer Batch get : %d", len(dict_prompt_sample['samples']))

              stop = s.evaluate_samples(dict_prompt_sample)  # CPU-intensive

              logging.info("Consumer Queue length: %d", q.qsize())
              if stop:
                  print("Stop is True")
                  should_stop.set()
                  break
      except Exception as e:
          logging.exception("Consumer exception: %s", e)
          should_stop.set()

  try:
      producer_thread = threading.Thread(target=producer, args=(s, iterations, logging))
      consumer_thread = threading.Thread(target=consumer, args=(s, logging))
      producer_thread.start()
      consumer_thread.start()
      producer_thread.join()
      consumer_thread.join()
      print('Threads have joined')
  except KeyboardInterrupt:
      logging.info("Keyboard interrupt. Stopping.")
      should_stop.set()
  finally:
      print("Backup everything")
      database.backup()


  # try:
  #   # This loop can be executed in parallel on remote sampler machines. As each
  #   # sampler enters an infinite loop, without parallelization only the first
  #   # sampler will do any work.
  #   while iterations != 0:
  #     # for s in samplers:
  #       samples = s.sample()
        
  #       should_stop = s.evaluate_samples(samples)
        
  #       if should_stop:
  #         break
  #     if should_stop:
  #       break
  #     if iterations > 0:
  #       iterations -= 1
  # except KeyboardInterrupt:
  #   logging.info("Keyboard interrupt. Stopping.")
  # database.backup()

  # for debug we can run only a sample.
  # sampler = samplers[0]
  # sampler.initialize_llm()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()
  # sampler.sample()

  ######################################################


@main.command()
@click.argument("db_file", type=click.File("rb"))
def ls(db_file):
  """List programs from a stored database (usually in data/backups/ )"""
  conf = config.Config(num_evaluators=1)

  # A bit silly way to list programs. This probably does not work if config has changed any way
  database = programs_database.ProgramsDatabase(
    conf.programs_database, None, "", identifier="")
  database.load(db_file)
  # import pdb; pdb.set_trace()
  progs = database.get_best_programs_per_island()
  print(f"Found {len(progs)} programs")
  for i, (prog, score) in enumerate(progs):
    print(f"{i}: Program with score {score}")
    print(prog)
    print("\n")


if __name__ == '__main__':
  main()
