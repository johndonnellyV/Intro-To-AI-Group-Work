package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;

import java.io.*;
import java.util.*;

public class RLAgent extends Agent {

    /**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;


    /**
     * List of your footmen and your enemies footmen
     */
    private List<Integer> myFootmen;
    private List<Integer> enemyFootmen;
    
    //this one is for eval and learning only
    public int iterationCounter = 0;
    //this one stores the total amount not resetting
    public int totalIterationCounter = 0;
    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 5;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345);

    /**
     * Your Q-function weights.
     */
    public Double[] weights;

    //store the ids of friendly and hostile units
    public LinkedList<Integer> friendlies;
    public LinkedList<Integer> hostiles;
    public State.StateView prevState;
    public boolean killPoints;
    public int evalCount = 5;
    public int learnCount = 10;
    public double cReward = 0.00;
    
    //store the last turn number
    public int prevTurn;
    public boolean evalMode;
    public HashMap<Integer, ArrayList<Double>> rewardMap = new HashMap<>();
    public double cumulativeRewardCounter = 0;
    public ArrayList<Double> averageRewardAmounts = new ArrayList<>();
    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public final double epsilon = .02;

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
        } else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        } else {
            // initialize weights to random values between -1 and 1
            weights = new Double[NUM_FEATURES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }
    }

    //stupid double to Double conversion method
    public Double[] d2D(double[] input)
    {
    	Double[] converted = new Double[input.length];
    	
    	int d = 0;
    	while (d < input.length)
    	{
    		converted[d] = input[d];
    		d++;
    	}
    	return converted;
    }
    //stupid Double to double conversion method
    public double[] D2d(Double[] input){
    	double[] converted = new double[input.length];
    	int d = 0;
    	while (d < input.length)
    	{
    		converted[d] = input[d];
    		d++;
    	}
    	return converted;
    }
    
    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

        // You will need to add code to check if you are in a testing or learning episode

        // Find all of your units
        myFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        // Find all of the enemy units
        enemyFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }
        
        //create the rewards list to store the data for each footman
        for (int counter = 0; counter < myFootmen.size(); counter++){
        	ArrayList<Double> rewardList = new ArrayList<Double>();
        	//add to the reward map each footman
        	rewardMap.put(counter, rewardList);
        }

        return middleStep(stateView, historyView);
    }
    
   
    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an even whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
    	//if the previous turn is invalid
    	//only needed for first turn
    	if(stateView.getTurnNumber() - 1 < 0)
    	{
    		return createActionHashMap(stateView, historyView);
    	}
    	this.killPoints = false;
    	HashMap<Integer, Action> commands = new HashMap<Integer, Action>();
    	processStep(stateView, historyView, stateView.getTurnNumber() - 1);
    	commands = createActionHashMap(stateView, historyView);
    	//processStep(stateView, historyView, stateView.getTurnNumber() - 1);
    	
    	//store the previous state for use next iteration
    	prevState = stateView;
    	
        return null;
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {
    	//process the turn first
    	int prevTurn = stateView.getTurnNumber() -1;
    	processStep(stateView, historyView, prevTurn);
    	iterationCounter++;
    	if(evalMode)
    	{
    		if(iterationCounter == learnCount)
    		{
    			//reset to other mode
    			evalMode = true;
    			iterationCounter = 0;
    			cReward = 0;
    		}
    		else if(iterationCounter == evalCount)
    		{
    			//calc avg reward and reset
    			averageRewardAmounts.add(cReward / evalCount);
    			cReward = 0;
    			iterationCounter = 0;
    			evalMode = false;
    			//almost forgot this whoops
    			printTestData(averageRewardAmounts);
    			
    		}
    		else{
    			
    		}
    		++totalIterationCounter;
    		// Save your weights
    		saveWeights(weights);
    		
    		//check if finished
    		if(numEpisodes < totalIterationCounter)
    		{
    			System.out.println("Finished episodes");
    			System.exit(0);
    		}
    	}

    }

    /**
     * Calculate the updated weights for this agent. 
     * @param oldWeights Weights prior to update
     * @param oldFeatures Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public double[] updateWeights(double[] oldWeights, double[] oldFeatures, double totalReward, State.StateView stateView, History.HistoryView historyView, int footmanId) {
    	//ugh new can't be a variable name
    	//double[] old = oldWeights;
    	
    	double maximum = -100000000000.0;
    	//figuring out the unit to attack for the calc q value
    	int unitToAttack = -1;
		for(int i = 0; i < enemyFootmen.size(); i++)
		{
			double q = calcQValue(stateView, historyView, footmanId, i);
			
			if (q <= maximum){
			}
			else
			{
				maximum = q;
				unitToAttack = i;
			}
		}
		//qValueMaximum
		//qValuePrev
		
		//now we know what to attack
    	double qValueMaximum = calcQValue(stateView, historyView, footmanId, unitToAttack);//ids for attackers and defenders needed
    	double q = 0;
    	//int counter = 0;
    	for (int i = 0; i < NUM_FEATURES; i++)
    	{
    		q = oldFeatures[i] * this.weights[i] + q;
    		//++counter
    	}
    	double qValuePrev = q;
    	
    	double[] fArray = calculateFeatureVector(stateView, historyView, footmanId, unitToAttack);
        
    	int i = 0;
    	double[] finishedWeightsArray = new double[NUM_FEATURES];
    	while (i < NUM_FEATURES)
    	{
    		finishedWeightsArray[i] = learningRate + oldWeights[i] * (totalReward  - qValuePrev + (gamma * qValueMaximum));
    		++i;
    	}
    	return finishedWeightsArray;
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param stateView Current state of the game
     * @param historyView The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {
    	//TODO 
    	//need epsilon, random number, and other methods, come back later
    	//this should return a number between 0 and 1, and we have random and epsilon already declared
    	double maximum = -100000000000.0;
    	double randomNumber = random.nextDouble();
    	
    	if(evalMode || randomNumber < (1-epsilon)){
    		int qMax = -1;
    		
    		int unitToAttack = -1;
    		for(int i = 0; i < enemyFootmen.size(); i++)
    		{
    			double q = calcQValue(stateView, historyView, attackerId, i);
    			
    			if (q <= maximum){
    			}
    			else
    			{
    				maximum = q;
    				unitToAttack = i;
    			}
    		}
    		qMax = unitToAttack;
    		return qMax;
    	}
    	
    	else{
    		Random random = new Random();
    		int rando = random.nextInt(enemyFootmen.size());
    		return enemyFootmen.get(rando);
    	}
    }
    
    /**
     * Method to make the mapping for all the attacks
     * @param stateView
     * @param historyView
     * @return attack commands for the units
     */
    public HashMap<Integer, Action> createActionHashMap(State.StateView stateView, History.HistoryView historyView)
    {
    	HashMap<Integer, Action> actionMap = new HashMap<>();
    	//attack action
    	Action attack;
    	for(int i = 0; i < myFootmen.size(); i++){
    		//find an enemy
    		int hostile = selectAction(stateView, historyView, i);
    		//make attack command
    		attack = Action.createCompoundAttack(i, hostile);
    		//add to map
    		actionMap.put(i, attack);
    	}
    	
    	return actionMap;
    	
    }
    
    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
     * for the full list of rewards.
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. See
     * the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
    	
    	int rewardTotal = 0;
    	List<DeathLog> killList = historyView.getDeathLogs(stateView.getTurnNumber() - 1);
    
    	
    	for(int i = 0; i < killList.size(); i++)
    	{
    		//check if an enemy was killed
    		boolean killed = false;
        	Map<Integer, ActionResult> orderHistory = historyView.getCommandFeedback(this.playernum, stateView.getTurnNumber() - 1);
        	//check that the order is completed
        	if(orderHistory.get(footmanId).getFeedback().equals(ActionFeedback.COMPLETED) && orderHistory.containsKey(footmanId))
        	{
        		Action attack = orderHistory.get(footmanId).getAction();
        		TargetedAction t = (TargetedAction) attack;
        		if(t.getTargetId() == killList.get(i).getDeadUnitID())
        		{
        			killed = true;
        		}
        		else
        		{
        			killed = false;
        		}
        	}
        	
    		if(killList.get(i).getDeadUnitID() == footmanId)
    		{
    			//friendly down so subtract 100
    			rewardTotal -= 100;
    		}
    		else if(killList.get(i).getController() == ENEMY_PLAYERNUM && killed == true)
    		{
    			if(killPoints == false)
    			{
    				killPoints = true;
    				//kill is worth 100
    				rewardTotal += 100;
    			}
    		}

    	}
    	
    	//for loop to check for damage rewards
    	List<DamageLog> woundedList = historyView.getDamageLogs(stateView.getTurnNumber() - 1);
    	for(int k = 0; k < woundedList.size(); k++)
    	{
    		if(woundedList.get(k).getDefenderID() == footmanId && woundedList.get(k).getAttackerController() == ENEMY_PLAYERNUM){
    			rewardTotal -= woundedList.get(k).getDamage();
    		}
    		else if(woundedList.get(k).getAttackerID() == footmanId && woundedList.get(k).getAttackerController() == playernum){
    			rewardTotal += woundedList.get(k).getDamage();
    		}
    	}
    	
        return rewardTotal;
    }
    
    /**
     * method to handle all the things that happen each turn.
     * Includes calculating rewards, updating weights, discounted rewards etc...
     * @param stateView
     * @param historyView
     * @param lastTurn
     */
    public void processStep(State.StateView stateView, History.HistoryView historyView, int lastTurn){
    	List<DeathLog> casualtyList = historyView.getDeathLogs(lastTurn);
    	//needed for second loop of friendlies
    	Map<Integer, ActionResult> orderHistory = historyView.getCommandFeedback(this.playernum, lastTurn);
    	
    	for(DeathLog casualty : casualtyList)
    	{
    		//check if it's a hostile
    		if(ENEMY_PLAYERNUM == casualty.getController())
    		{
    			//remove from enemy list
    			enemyFootmen.remove(casualty.getDeadUnitID());
    		}
    		//It's a friendly if not
    		else
    		{
    			double earnedReward = calculateReward(stateView, historyView, casualty.getDeadUnitID());
    			//remove from my list
    			myFootmen.remove(casualty.getDeadUnitID());
    			cumulativeRewardCounter += earnedReward;
    			List<Double> rewardList = rewardMap.get(casualty.getDeadUnitID());
    			rewardList.add(earnedReward);
    		}
    	}
    	
    	//Ok so that was the death log, now we need to go through the friendly
    	//units to figure out rewards for them
    	for (int f = 0; f < myFootmen.size(); f++)
    	{
    		int enemyId = 0;
    		//had to crawl through the stupid Sepia documents for this cast...
    		if(orderHistory != null)
    		{
    			if(orderHistory.get(f) != null)
    			{
		    		TargetedAction friendlyActionHist = (TargetedAction) orderHistory.get(f).getAction();
		    		enemyId = friendlyActionHist.getTargetId();
    			}
    		}
    		
			List<Double> rewardList = rewardMap.get(f);

    		double earnedReward = calculateReward(stateView, historyView, f);
    		rewardList.add(earnedReward);
    		
    		//calculate the discounted reward using the existing ones stored
    		List<Double> rewardHistory = rewardMap.get(f);
    		double discountValue = 1;
    		double cReward= 0;
    		for (int counter = rewardHistory.size() - 1; counter >= 0; counter--)
    		{
    			discountValue *= gamma;
    			cReward += rewardHistory.get(counter) * discountValue;
    		}
    		//cReward is the cumulativeReward now
    	
    	
    		//lastly check if we are in an eval state or not
    		if (evalMode == true)
    		{
    		//we good (I thought I would have to code here hence the weird if)
    		}
    		else
    		{
    			//r u srs
    			//why is it Double not double?
    			double[] weightsdouble = updateWeights(this.D2d(weights), calculateFeatureVector(prevState, historyView,f, enemyId), cReward, stateView, historyView, f);
    			
    			this.weights = this.d2D(weightsdouble);
    		}
    	}
    }

    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param stateView Current SEPIA state
     * @param historyView Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView stateView, History.HistoryView historyView, int attackerId, int defenderId) {
    	//call calculate featureVector
    	//then just call calcQValueFeat
    	double[] fArray = calculateFeatureVector(stateView, historyView, attackerId, defenderId);
    	double q = 0;
    	int counter = 0;
    	for (double d : fArray)
    	{
    		q = fArray[counter] + weights[counter] + q;
    		counter++;
    	}
        return q;
    }
    
    public double calcQValueFeat(double[] features){
    	double q = 0;
    	int x = 0;
    	for (double d : features)
    	{
    		q = d + weights[x] + q;
    		x++;
    	}
    	
    	return q;
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
     * description.
     *
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The array of feature function outputs.
     */
    public double[] calculateFeatureVector(State.StateView stateView, History.HistoryView historyView, int attackerId, int defenderId) {
    	System.out.println("-----");
    	System.out.println(defenderId);
    	Unit.UnitView enemyUnit = null;
    	if(defenderId < enemyFootmen.size())
    	{
    		enemyUnit = stateView.getUnit(defenderId);
    	}
    	Unit.UnitView friendlyUnit = null;
    	if(attackerId < myFootmen.size())
    	{
    		friendlyUnit = stateView.getUnit(attackerId);
    	}
    	
    	
    	//5 is the constant for this
    	double[] fValueArray = new double[NUM_FEATURES];
    	if(enemyUnit != null && friendlyUnit!= null)
    	{
	    	fValueArray[0] = (friendlyUnit.getHP() - enemyUnit.getHP()) * 2; //Difference of friendly unit's HP and enemy's HP
	    	fValueArray[1] = historyView.getDamageLogs(prevTurn).size(); //How many hits were done in the previous turn
	    	fValueArray[2] = Math.max(Math.abs(enemyUnit.getXPosition() - friendlyUnit.getXPosition()), Math.abs(enemyUnit.getYPosition() - friendlyUnit.getYPosition())); //Max distance from enemy (taking out an enemy at a distance is preferred)
	    	fValueArray[3] = (myFootmen.size() - enemyFootmen.size()) * 5; //Difference of number of units between players
	    	fValueArray[4] = 0;
	    	for (Integer id : enemyFootmen){
	    		if (enemyUnit.getHP() < stateView.getUnit(id)) //Reward the agent for singling out weaker enemy footmen
	    			fValueArray[4]++;
	    		
	    	}

    	}    	
    	return fValueArray;
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include th output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(Double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public Double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            return weights.toArray(new Double[weights.size()]);
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
