#!/usr/bin/env node
import { runDoctorCli } from './doctor-cli.js';
import { runTraceCli } from './trace-cli.js';
import { runSimulateCli } from './simulate-cli.js';

const argv = process.argv.slice(2);
const run = argv[0] === 'simulate' ? runSimulateCli(argv) : argv[0] === 'trace' ? runTraceCli(argv) : runDoctorCli(argv);
void run.then((code) => { process.exitCode = code; });
